from collections import defaultdict
import logging
from copy import deepcopy

import numpy as np
import os
import pickle
import networkx as nx
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from egraph_model import BaseEGraph
from math import ceil

import torch_sparse
from torch_sparse import SparseTensor, t
from torch_sparse import max as spmax
from torch_sparse import sum as spsum
from torch_sparse import min as spmin
from torch_geometric.utils import softmax
from exp_m import sparse_expm, dense_expm
from tqdm import tqdm
# from pytorch_memlab import LineProfiler, profile

spmm = torch_sparse.matmul


def sparse_gumbel_softmax(src,
                          row,
                          col,
                          shape,
                          tau=1,
                          hard=False,
                          dim=-1,
                          return_format='torch_sparse'):
    # Generate Gumbel noise for non-zero elements in logits
    gumbel_noise = -torch.empty_like(src).exponential_().log()
    perturbed_logits = (src + gumbel_noise) / tau
    # Apply sparse softmax to the perturbed logits
    if dim == 1:
        y_soft = softmax(perturbed_logits.flatten(), index=row)
    elif dim == 0:
        y_soft = softmax(perturbed_logits.flatten(), index=col)
    else:
        raise ValueError

    # the shape [BM, BN], implicitly reshape it to [BN, BM] here
    if return_format == 'torch_sparse':
        ret = SparseTensor(row=col, col=row, value=y_soft, sparse_sizes=shape)
    elif return_format == 'torch':
        ret = torch.sparse_coo_tensor(indices=torch.stack([col, row]),
                                      values=y_soft,
                                      size=shape)

    if hard:
        # TODO: fix this branch for torch return_format
        _, row_count = torch.unique_consecutive(row, return_counts=True)
        max_per_row = spmax(ret, dim=0)
        max_per_col = torch.repeat_interleave(max_per_row, row_count)
        max_mask = (y_soft == max_per_col)
        hard_ret = SparseTensor(row=col[max_mask],
                                col=row[max_mask],
                                value=torch.ones(max_mask.sum(),
                                                 device=src.device),
                                sparse_sizes=shape)
        neg_ret = SparseTensor(row=col,
                               col=row,
                               value=-y_soft,
                               sparse_sizes=shape)
        ret = hard_ret + neg_ret.detach() + ret
    return ret


def sparse_softmax(src, row, col, shape, dim=-1):
    if dim == 1:
        y_soft = softmax(src.flatten(), index=row)
    elif dim == 0:
        y_soft = softmax(src.flatten(), index=col)
    else:
        raise ValueError
    ret = SparseTensor(row=col, col=row, value=y_soft, sparse_sizes=shape)
    return ret


def sparse_normalize(src, row, col, shape, dim=-1):
    src = torch.sigmoid(src.flatten())

    temp_sparse = SparseTensor(row=col, col=row, value=src, sparse_sizes=shape)
    _, row_count = torch.unique_consecutive(row, return_counts=True)
    sum_per_row = spsum(temp_sparse, dim=0)
    sum_per_col = torch.repeat_interleave(sum_per_row, row_count)
    y_soft = src / sum_per_col

    ret = SparseTensor(row=col, col=row, value=y_soft, sparse_sizes=shape)
    return ret


class SparseEGraph(BaseEGraph):

    def __init__(
        self,
        input_file,
        batch_size=None,
        hidden_dim=32,
        num_attributes=5,
        gumbel_tau=1,
        dropout=0.0,
        eps=0.5,
        soft=False,
        embedding_type='lookup',
        aggregate_type='mean',
        logit2prob_func='gumbel_softmax',
        assumtion='hybrid',
        device='cuda',
        load_cost=False,
        gpus=1,
        filter_cycles=False,
        greedy_ini=False,
        compress=True,
        drop_self_loops=True,
    ):
        """
        Notations used in the comment of this class:
        N: number of nodes
        M: number of classes
        H: hidden dimension
        """
        self.batch_size = batch_size
        self.filter_cycles = filter_cycles
        super().__init__(input_file,
                         hidden_dim,
                         num_attributes,
                         gumbel_tau,
                         dropout,
                         eps,
                         batch_size,
                         soft,
                         embedding_type,
                         aggregate_type,
                         device,
                         load_cost,
                         greedy_ini,
                         compress=compress,
                         drop_self_loops=drop_self_loops)

        self.gpus = max(gpus, 1)
        assert self.batch_size % self.gpus == 0

        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        self.logit2prob_func = logit2prob_func
        self.assumption = assumtion

        # [N, M], the classes pointed by the node
        self.node2classT = t(self.node2class)

        self.cyclic_loss = torch.tensor([0.0], device=self.device)
        self.cyclic_count = 0
        self.set_index()
        self.known_cycles = []

    @torch.no_grad()
    def set_index(self):
        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        # [N], the class consists of the node
        # class_per_node_backup = self.class_adj.nonzero()[:, 0]
        class_per_node = self.class2node.storage._row.clone()
        self.class_per_node = nn.Parameter(class_per_node.repeat(B).to(
            self.device),
                                           requires_grad=False)
        self.batch_per_node = torch.repeat_interleave(
            torch.arange(B, device=self.device), N)
        self.node_per_node = torch.arange(N, device=self.device).repeat(B)

        self.index0 = nn.Parameter(self.batch_per_node * M +
                                   self.class_per_node,
                                   requires_grad=False)
        self.index1 = nn.Parameter(self.batch_per_node * N +
                                   self.node_per_node,
                                   requires_grad=False)

    def find_cycles(self, batch_choose_enodes, cycle_info=False):

        def cycle_dfs(class_id):
            node_id = cls_node_dict[class_id]
            child_classes = self.enodes[node_id].eclass_id

            if status[class_id] == "Done":
                return
            elif status[class_id] == "Doing":
                if cycle_info:
                    cycle_start_index = stack.index((class_id, node_id))
                    class_cycle = [c[0] for c in stack[cycle_start_index:]]
                    node_cycle = [c[1] for c in stack[cycle_start_index:]]
                    cycles.append((class_cycle, node_cycle))
                else:
                    cycles.append(class_id)
                return

            status[class_id] = "Doing"
            stack.append((class_id, node_id))
            for child in child_classes:
                cycle_dfs(child)

            status[class_id] = "Done"
            stack.pop()

        # all use int index for eclass and enode
        batch_cycles = []
        batch_cycle_num = []
        for batch in range(self.batch_size):
            enodes = torch.where(batch_choose_enodes[batch])[0].tolist()
            if len(enodes) == 0:
                batch_cycles.append([])
                batch_cycle_num.append(0)
                continue

            # map class_id to the chosen enode_id
            cls_node_dict = {
                self.enodes[enode].belong_eclass_id: enode
                for enode in np.atleast_1d(enodes)
            }

            status = defaultdict(lambda: "Todo")
            cycles = []
            stack = []
            for root in self.root:
                cycle_dfs(root)
            batch_cycles.append(cycles)
            batch_cycle_num.append(len(cycles))
        return batch_cycle_num, batch_cycles

    def set_root(self):
        if hasattr(self, 'vector_root'):
            return self.vector_root
        if hasattr(self, 'root'):
            if isinstance(self.root, int):
                self.root = [self.root]
            elif isinstance(self.root, list):
                self.root = [self.class_mapping[r] for r in self.root]
            else:
                raise NotImplementedError
            self.vector_root = torch.zeros(len(self.eclasses),
                                           dtype=torch.bool,
                                           device=self.device)
            self.vector_root[self.root] = 1
        else:
            self.vector_root = torch.ones(len(self.eclasses),
                                          dtype=torch.bool,
                                          device=self.device)
            self.vector_root[self.node2class.storage._col] = 0
            self.root = self.vector_root.nonzero().squeeze().tolist()
            self.root = np.atleast_1d(self.root)
        return self.vector_root

    @torch.no_grad()
    def step(self):
        # in-place hack for nn.DataParallel
        if self.training:
            self.gumbel_tau += self.temperature_schedule[
                self.step_count] - self.gumbel_tau
            self.step_count += 1
        self.cyclic_loss.zero_()

    def init_params(self):
        self.enode_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                          out_features=self.hidden_dim)
        self.output_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                           out_features=1)

        self.activation = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim), torch.nn.ReLU())
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.set_attr(self.num_attributes)

    def forward_embedding(self, embedding):
        if embedding.shape[-1] == 1:
            return embedding.squeeze(-1)
        else:
            return super().forward_embedding(embedding)

    def dense(self, enode_embedding):
        # enode_embedding: [n, hidden_dim]
        # return: [n, hidden_dim]
        return self.enode_proj(enode_embedding)

    @torch.no_grad()
    def select_batch(self, new_batch_size, enodes):
        loss = self.cost_per_node * enodes
        loss = loss.sum(dim=1)
        best_batches = loss.topk(new_batch_size).indices

        self.batch_size = new_batch_size
        self.embedding = nn.Parameter(self.embedding[best_batches],
                                      requires_grad=True)
        self.set_index()

    def forloop_spmm(self, A, B, m, n, b):
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        row = indices[0]
        col = indices[1]
        nnz = values.numel() // b

        new_indices = []
        new_values = []

        for i in range(b):
            sub_row = row[i * nnz:(i + 1) * nnz] % m
            sub_col = col[i * nnz:(i + 1) * nnz] % n
            A_sparse = torch.sparse_coo_tensor(
                indices=torch.stack([sub_row, sub_col]),
                values=values[i * nnz:(i + 1) * nnz],
                size=(m, n))

            C = torch.sparse.mm(A_sparse, B).to_sparse_coo()
            new_indices.append(C.indices() + i * m)
            new_values.append(C.values())

        new_indices = torch.cat(new_indices, dim=1)
        new_values = torch.cat(new_values, dim=0)
        sparse_C = torch.sparse_coo_tensor(indices=new_indices,
                                           values=new_values,
                                           size=(m * b, m * b))
        sparse_C = sparse_C.coalesce()
        return sparse_C

    def forloop_spgemm(self, A, B, m, n, b):
        B = B.to_torch_sparse_coo_tensor()
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        row = indices[0]
        col = indices[1]
        nnz = values.numel() // b

        new_indices = []
        new_values = []

        for i in range(b):
            sub_row = row[i * nnz:(i + 1) * nnz] % m
            sub_col = col[i * nnz:(i + 1) * nnz] % n
            A_sparse = torch.sparse_coo_tensor(
                indices=torch.stack([sub_row, sub_col]),
                values=values[i * nnz:(i + 1) * nnz],
                size=(m, n))

            C = A_sparse @ B
            new_indices.append(C.indices() + i * m)
            new_values.append(C.values())

        new_indices = torch.cat(new_indices, dim=1)
        new_values = torch.cat(new_values, dim=0)
        sparse_C = torch.sparse_coo_tensor(indices=new_indices,
                                           values=new_values,
                                           size=(m * b, m * b))
        sparse_C = sparse_C.coalesce()
        return sparse_C

    # @profile
    def sample_v2(self, embedding, hard=False):
        B, M, N = self.batch_size // self.gpus, len(self.eclasses), len(
            self.enodes)
        eps = 1e-10
        device = embedding.device
        node_prob = torch.zeros((B, N), device=device, requires_grad=False)

        node_logits = self.forward_embedding(embedding)  # [B, N]

        probs_class2node = sparse_gumbel_softmax(node_logits,
                                                 row=self.index0[:B * N],
                                                 col=self.index1[:B * N],
                                                 shape=(B * N, B * M),
                                                 dim=1,
                                                 tau=self.gumbel_tau,
                                                 hard=hard,
                                                 return_format='torch_sparse')

        probs_class2node_clone = probs_class2node.clone()
        self.probs_class2node = probs_class2node
        if self.filter_cycles:
            # reduce the batch dimension on the class dimension
            reshaped_probs_class2node = probs_class2node.clone()
            reshaped_probs_class2node.storage._col %= M
            reshaped_probs_class2node.storage._sparse_sizes = (B * N, M)

            self.compute_cyclic_loss2(reshaped_probs_class2node)

        row = probs_class2node.storage._row
        col = probs_class2node.storage._col
        values = probs_class2node.storage._value
        probs_class2node = torch.sparse_coo_tensor(indices=torch.stack(
            [col, row]),
                                                   values=values,
                                                   size=(B * M, B * N))

        # probs_class2node2 = torch.sparse_coo_tensor(indices=torch.stack(
        #     [col, row % N]),
        #                                             values=values,
        #                                             size=(B * M, N))

        if not hasattr(self, 'batch_node2class'):
            self.node2class = self.node2class.to(device)
            row = self.node2class.storage._row
            col = self.node2class.storage._col
            value = self.node2class.storage._value
            nnz = row.numel()
            batch_index = torch.arange(B, device=device).repeat_interleave(nnz)
            self.batch_node2class = torch.sparse_coo_tensor(
                indices=torch.stack([
                    row.repeat(B) + batch_index * N,
                    col.repeat(B) + batch_index * M
                ]),
                values=value.repeat(B),
                size=(B * N, B * M))

        # [BM, BN] @ [BN, BM] -> [BM, BM]
        # c2c = probs_class2node @ self.batch_node2class
        c2c = probs_class2node @ self.batch_node2class.to(device)
        # torch.save(probs_class2node, 'c2n.pt')
        # torch.save(self.batch_node2class.to(device), 'n2c.pt')
        # breakpoint()
        # c2c = self.forloop_spmm(probs_class2node, self.node_adj.float(), M, N, B)
        # c2c = self.forloop_spgemm(probs_class2node, self.node2class, M, N, B)

        # c2c = probs_class2node2 @ self.node2class.to_torch_sparse_coo_tensor()
        indices = c2c.indices()
        # indices[1] += indices[0] // M * M

        class_prob = torch.zeros((B * M), device=device, requires_grad=False)
        max_norm = 0
        for i in range(M):
            if i == 0:
                cur_class_prob = self.set_root().float().to(device)
                cur_class_prob = cur_class_prob.unsqueeze(0).expand(
                    B, M).flatten()
            else:
                # # [B, M] -> [BM, 1]
                # class_prob = class_prob.flatten()

                value = c2c.values()
                value = value * class_prob[indices[0]]

                if self.assumption in ['correlated', 'hybrid']:
                    # assume all eclasses are correlated
                    cor_cur_class_prob = SparseTensor(row=indices[0],
                                                      col=indices[1],
                                                      value=value,
                                                      sparse_sizes=(B * M,
                                                                    B * M))
                    cor_cur_class_prob = cor_cur_class_prob.max(dim=0)

                if self.assumption in [
                        'independent', 'hybrid', 'neg_correlated'
                ]:
                    # assume all eclasses are independent
                    value = torch.log((1 - value).clamp(eps, 1.0))
                    ind_cur_class_prob = SparseTensor(row=indices[0],
                                                      col=indices[1],
                                                      value=value,
                                                      sparse_sizes=(B * M,
                                                                    B * M))
                    ind_cur_class_prob = ind_cur_class_prob.sum(dim=0)
                    ind_cur_class_prob = ind_cur_class_prob.to_dense()
                    ind_cur_class_prob = 1 - torch.exp(ind_cur_class_prob)

                if self.assumption == 'neg_correlated':
                    cor_cur_class_prob = SparseTensor(row=indices[0],
                                                      col=indices[1],
                                                      value=value,
                                                      sparse_sizes=(B * M,
                                                                    B * M))
                    cor_cur_class_prob = cor_cur_class_prob.sum(dim=0)

                if self.assumption == 'correlated':
                    cur_class_prob = cor_cur_class_prob.to_dense()
                elif self.assumption == 'independent':
                    cur_class_prob = ind_cur_class_prob
                elif self.assumption == 'neg_correlated':
                    cur_class_prob = cor_cur_class_prob
                    cur_class_prob[cur_class_prob > 1] = ind_cur_class_prob[
                        cur_class_prob > 1]
                elif self.assumption == 'hybrid':
                    cur_class_prob = (cor_cur_class_prob.to_dense() +
                                      ind_cur_class_prob) / 2
                else:
                    raise NotImplementedError

            class_prob = torch.maximum(class_prob, cur_class_prob)
            cur_norm = class_prob.norm().item()
            if abs(cur_norm - max_norm) / max(max_norm, 1) < 1e-5:
                logging.info(f'converged at {i} iter')
                break
            else:
                max_norm = max(cur_norm, max_norm)

        # breakpoint()
        # [BN, BM] @ [BM, 1] -> [BN, 1] -> [B, N]
        node_prob = spmm(probs_class2node_clone, class_prob.view(-1, 1))
        torch.cuda.empty_cache()
        return node_prob.view(B, N), self.cyclic_loss.unsqueeze(0)

    # @profile
    def sample(self, embedding, hard=False):
        B, M, N = self.batch_size // self.gpus, len(self.eclasses), len(
            self.enodes)
        eps = 1e-10

        class_prob = torch.zeros((B, M),
                                 device=embedding.device,
                                 requires_grad=False)
        node_prob = torch.zeros((B, N),
                                device=embedding.device,
                                requires_grad=False)

        node_logits = self.forward_embedding(embedding)  # [B, N]

        if self.logit2prob_func == 'gumbel_softmax':
            probs_class2node = sparse_gumbel_softmax(node_logits,
                                                     row=self.index0[:B * N],
                                                     col=self.index1[:B * N],
                                                     shape=(B * N, B * M),
                                                     dim=1,
                                                     tau=self.gumbel_tau,
                                                     hard=hard)
        else:
            if self.logit2prob_func == 'softmax':
                logits2prob_func = sparse_softmax
            elif self.logit2prob_func == 'normalize':
                logits2prob_func = sparse_normalize
            probs_class2node = logits2prob_func(
                node_logits,
                row=self.index0[:B * N],
                col=self.index1[:B * N],
                shape=(B * N, B * M),
                dim=1,
            )
        if self.filter_cycles:
            # reduce the batch dimension on the class dimension
            reshaped_probs_class2node = probs_class2node.clone()
            reshaped_probs_class2node.storage._col %= M
            reshaped_probs_class2node.storage._sparse_sizes = (B * N, M)

            self.compute_cyclic_loss2(reshaped_probs_class2node)

        self.node2class = self.node2class.to(embedding.device)
        self.node2classT = self.node2classT.to(embedding.device)

        probs = []
        for i, classes in enumerate(self.ordered_classes):
            if i == 0:
                # root classes
                cur_class_prob = classes.detach().clone()
                cur_class_prob = cur_class_prob.unsqueeze(0).expand(B, M)
            else:
                # [N, M] @ [M] -> [N]
                nodes = spmm(self.node2class,
                             classes.unsqueeze(-1)).squeeze().bool().float()

                # assume all nodes are independent
                # [B, N] * [N] -> [B, N]
                cur_node_prob = 1 - node_prob
                log_node_prob = torch.log(cur_node_prob + eps) * nodes
                # [B, N] @ [N, M] -> [B, M]
                # log_class_prob = spmm(self.node2class.T, log_node_prob.T).T
                log_class_prob = spmm(self.node2classT, log_node_prob.T).T
                # [B, M] * [M] -> [B, M]
                cur_class_prob = (1 - torch.exp(log_class_prob)) * classes

                # assume all nodes are fully correlated
                # # [B, N] * [N] -> [B, N]
                # exp_node_prob = torch.exp(cur_node_prob) * nodes
                # # # [B, N] @ [N, M] -> [B, M]
                # exp_class_prob = spmm(self.node2classT, exp_node_prob.T).T * classes
                # # # [B, M] @ [M, N] -> [B, N]
                # softmax_dom = spmm(self.node2class, exp_class_prob.T).T
                # normalized_node_prob = (cur_node_prob * nodes) * (exp_node_prob / softmax_dom.clamp(min=torch.e))
                # # [B, N] @ [N, M] -> [B, M]
                # cur_class_prob = spmm(self.node2classT, normalized_node_prob.T).T
                # cur_class_prob = cur_class_prob * classes

            class_prob += cur_class_prob
            # the following is a more efficient way to compute node_prob
            # it is because the batched sparse matrix multiplication is
            # not supported by PyTorch yet
            # [BN, BM] @ [B, M] -> [BN, BM] @ [BM, 1] -> [BN, 1] -> [B, N]
            cur_node_prob = spmm(probs_class2node,
                                 cur_class_prob.reshape(-1, 1))
            cur_node_prob = cur_node_prob.view(B, -1)
            node_prob += cur_node_prob

        return node_prob

    def inference_sample(self, embedding):

        def update_inference_class2node(node_logits):
            # Switch to deterministic sample during inference using max
            logits_class2node = SparseTensor(row=self.index0,
                                             col=self.index1,
                                             value=node_logits,
                                             sparse_sizes=(B * M, B * N))
            _, row_count = torch.unique_consecutive(self.index0,
                                                    return_counts=True)
            max_per_col = spmax(logits_class2node, dim=1)
            max_per_row = torch.repeat_interleave(max_per_col, row_count)
            max_mask = (node_logits == max_per_row)
            # implicitly reshape [BM, BN] -> [BN, BM]
            class2node = SparseTensor(row=self.index1[max_mask],
                                      col=self.index0[max_mask],
                                      value=torch.ones(
                                          max_mask.sum(),
                                          device=embedding.device),
                                      sparse_sizes=(B * N, B * M))
            return class2node, row_count

        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        # root_classes = self.ordered_classes[0]
        root_classes = self.set_root()
        # [M] -> [B, M]
        root_classes = root_classes.repeat(B, 1).bool()
        visited_classes = torch.zeros(B,
                                      M,
                                      dtype=torch.bool,
                                      device=embedding.device)
        visited_nodes = torch.zeros(B,
                                    N,
                                    dtype=torch.bool,
                                    device=embedding.device)

        node_logits = self.forward_embedding(embedding)  # [B, N]

        # [BM, BN]
        node_logits = node_logits.flatten()
        class2node, row_count = update_inference_class2node(node_logits)
        branch = (row_count > 1).view(B, -1)

        # Switch to random sample during inference
        # class2node = sparse_gumbel_softmax(node_logits,
        #                                    row=self.index0[:B * N],
        #                                    col=self.index1[:B * N],
        #                                    shape=(B * N, B * M),
        #                                    dim=1,
        #                                    tau=1,
        #                                    hard=True)

        cycled_batch = torch.zeros(B,
                                   dtype=torch.bool,
                                   device=embedding.device)
        branch_enodes = torch.zeros(B,
                                    N,
                                    dtype=torch.bool,
                                    device=embedding.device)
        mask_enodes = torch.zeros(B,
                                  N,
                                  dtype=torch.bool,
                                  device=embedding.device)

        if not hasattr(self, 'batch_class2node'):
            self.batch_class2node = SparseTensor(row=self.index1,
                                                 col=self.index0,
                                                 value=torch.ones(
                                                     self.index0.numel(),
                                                     device=embedding.device),
                                                 sparse_sizes=(B * N, B * M))
            self.batch_class2nodeT = self.batch_class2node.t()

        while True:
            visited_classes |= root_classes
            # [BN, BM] @ [B, M] -> [BN, BM] @ [BM, 1] -> [BN, 1] -> [B, N]
            root_nodes = spmm(class2node, root_classes.float().reshape(-1, 1))
            root_nodes = root_nodes.view(B, -1)
            visited_nodes |= root_nodes.bool()

            branch_eclasses = branch & root_classes
            if branch_eclasses.any():
                branch_enodes[branch_eclasses.any(dim=1)] = 0
                update = spmm(class2node,
                              branch_eclasses.float().view(-1, 1)).view(B, -1)
                # if all_enodes.sum() == (branch_enodes & all_enodes).sum():
                branch_enodes_per_class = spmm(self.batch_class2nodeT,
                                               mask_enodes.float().view(
                                                   -1, 1)).flatten()
                # when all enodes are tried for an eclass, reset this branch
                # TODO: debug the following line
                branch &= (branch_enodes_per_class != row_count).reshape(B, -1)
                # if (branch_enodes_per_class == row_count).sum() > 0:
                #     pass
                branch_enodes |= update.bool()
                mask_enodes |= update.bool()

            # [B, N] @ [N, M] -> [B, M]
            root_classes = spmm(self.node2classT, root_nodes.T).T
            root_classes = root_classes.bool()
            if (root_classes
                    & visited_classes).any() and self.step_count > 1000:
                # cycle detected
                cycled_batch |= (root_classes & visited_classes).any(dim=1)
                if cycled_batch.all():
                    # restart when all batches are cycled
                    logging.info(
                        f'node logits norm: {node_logits.norm()}, restart')
                    logging.info(
                        f'branch: {torch.where(branch_enodes.view(B, -1))}')
                    min_logits = node_logits.min()
                    node_logits[branch_enodes.flatten()] = min_logits
                    class2node, _ = update_inference_class2node(node_logits)

                    root_classes = self.set_root()
                    root_classes = root_classes.repeat(B, 1).bool()
                    visited_nodes.zero_()
                    visited_classes.zero_()
                    branch_eclasses.zero_()
                    branch_enodes.zero_()
            l = torch.where(visited_nodes)[1]

            root_classes &= (~visited_classes)

            if not root_classes.any():
                break

        return visited_nodes

    def hard_sample(self, embedding):
        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        root_classes = self.ordered_classes[0]
        # [M] -> [B, M]
        root_classes = root_classes.repeat(B, 1).bool()
        visited_classes = torch.zeros((B, M),
                                      dtype=torch.float,
                                      device=embedding.device)
        visited_nodes = torch.zeros((B, N),
                                    dtype=torch.float,
                                    device=embedding.device)

        node_logits = self.forward_embedding(embedding)  # [B, N]

        # [BM, BN]
        node_logits = node_logits.flatten()
        class2node = sparse_gumbel_softmax(node_logits,
                                           row=self.index0[:B * N],
                                           col=self.index1[:B * N],
                                           shape=(B * N, B * M),
                                           dim=1,
                                           tau=self.gumbel_tau,
                                           hard=True)

        while True:
            visited_classes += root_classes
            # [BN, BM] @ [B, M] -> [BN, BM] @ [BM, 1] -> [BN, 1] -> [B, N]
            root_nodes = spmm(class2node, root_classes.float().reshape(-1, 1))
            root_nodes = root_nodes.view(B, -1)
            visited_nodes += root_nodes

            # [B, N] @ [N, M] -> [B, M]
            root_classes = spmm(self.node2classT, root_nodes.T).T
            root_classes *= (1 - visited_classes)
            # use hardtanh as a hard clamp with STE
            root_classes = F.hardtanh(root_classes, min_val=0, max_val=1)
            if root_classes.sum() == 0:
                break
        return visited_nodes

    def find_scc(self, n2n):
        scc_labels = scipy.sparse.csgraph.connected_components(
            n2n, connection='strong')
        return scc_labels

    def class2edge_mask(self, scc_mask):
        row_mask = torch.zeros(self.nnz, dtype=torch.bool, device=self.device)
        col_mask = torch.zeros(self.nnz, dtype=torch.bool, device=self.device)
        n_range = np.arange(len(self.eclasses))
        for node in n_range[scc_mask]:
            row_mask |= (self.row == node)
            col_mask |= (self.col == node)
        edge_mask = row_mask & col_mask
        # if edge_mask.sum() >= 0:
        #     print(f'scc: {edge_mask.sum()}')
        return edge_mask

    def get_edge_masks(self, n2n):
        N = n2n.shape[0]
        self.nnz = n2n._nnz()
        self.row = n2n.coalesce().indices()[0]
        self.col = n2n.coalesce().indices()[1]
        values = n2n.coalesce().values()
        self.edge_masks = []
        self.components = []

        scipy_n2n = scipy.sparse.coo_matrix(
            (values.cpu().bool().numpy(),
             (self.row.cpu().numpy(), self.col.cpu().numpy())),
            shape=(N, N))

        n_scc, scc_labels = self.find_scc(scipy_n2n)
        for label in range(n_scc):
            scc_mask = (scc_labels == label)
            degree = scc_mask.sum()
            edge_mask = self.class2edge_mask(scc_mask)
            if degree > 1:
                self.edge_masks.append((degree, edge_mask))
            self.components.append(np.where(scc_mask)[0].tolist())

        # self.components = [set(c) for c in self.components]

    def compute_cyclic_loss2(self,
                             probs_class2node,
                             method='sparse_reduce_scc'):
        assert method in ['sparse_reduce', 'sparse_reduce_scc', 'dense']

        B, M, N = self.batch_size // self.gpus, len(self.eclasses), len(
            self.enodes)
        if method == 'dense':
            dense_probs = probs_class2node.to_dense().reshape(B, M, N)
            dense_node2class = self.node2class.to_dense()
            batched_c2c = dense_probs @ dense_node2class
            exp = torch.matrix_exp(batched_c2c)
            self.cyclic_loss = torch.einsum('bmm->b', exp).mean() - M
            print(f'cyclic loss = {self.cyclic_loss}')
            return

        values = probs_class2node.storage.value()
        values = values.view(B, -1).mean(dim=0)
        row = probs_class2node.storage.row()
        col = probs_class2node.storage.col()
        nnz = row.numel() // B
        device = values.device

        probs_class2node = torch.sparse_coo_tensor(
            torch.stack([row[:nnz], col[:nnz]], dim=0), values, (N, M))
        c2c = self.node2classT.to_torch_sparse_coo_tensor().to(
            device) @ probs_class2node

        if method == 'sparse_reduce':
            assert self.gpus == 1
            self.cyclic_loss = (sparse_expm(c2c) - M)
            self.cyclic_loss = self.cyclic_loss.mean()
        elif method == 'sparse_reduce_scc':
            cyclic_loss = 0
            if not hasattr(self, 'edge_masks'):
                self.get_edge_masks(c2c.clone().detach())

            values = c2c.coalesce().values()
            indices = c2c.coalesce().indices()
            for degree, edge_mask in self.edge_masks:
                if degree > 1:
                    _, reverse_index = torch.unique(indices[:, edge_mask],
                                                    return_inverse=True)
                    # multiply a constant to increase the numerical stability
                    if degree > 5000:
                        value = values[edge_mask] * 3
                    else:
                        value = values[edge_mask]

                    scc = torch.sparse_coo_tensor(reverse_index, value,
                                                  (degree, degree))
                    expm = sparse_expm(scc)
                    if degree > 5000:
                        expm *= 1e-2
                    cyclic_loss += expm
            if cyclic_loss != 0:
                self.cyclic_loss = cyclic_loss
            else:
                logging.info('no cycles detected in training')

    def compute_loss(self,
                     enodes,
                     cyclic_loss=0,
                     backward=True,
                     optim_goal='sum',
                     debug=False,
                     verbose=False,
                     cycle_info=False):
        assert optim_goal == 'sum'

        # This paprameter should be problem specific
        # will change to input args later
        penalty = 100
        raw_enodes = enodes @ self.nodes2raw

        # add quadratic cost if exists
        if hasattr(self, 'quad_cost_mat'):
            loss = self.quad_cost(raw_enodes)
        elif hasattr(self, 'mlp'):
            loss = self.mlp_cost(raw_enodes)
        else:
            loss = self.linear_cost(raw_enodes)

        if self.training:
            loss = loss.mean()

            logging.info(f'Training loss mean: {loss.mean().item():.4f}')
            if self.filter_cycles and self.cyclic_count > 0 and (cyclic_loss
                                                                 > 0).any():
                cyclic_loss = cyclic_loss.mean()
                if self.cyclic_count < 5:
                    cyclic_coef = self.cyclic_count**2
                else:
                    cyclic_coef = 2**self.cyclic_count
                cyclic_coef *= loss.abs().item() * self.reg
                loss += cyclic_loss * cyclic_coef

                logging.info(f'Cyclic loss = {cyclic_loss}')
            logging.info(f'Cyclic count = {self.cyclic_count}')

        else:
            best_batch = loss.argmin()
            if self.filter_cycles:
                cycle_num, cycles = self.find_cycles(enodes, cycle_info)
                cycle_num = torch.tensor(cycle_num,
                                         device=self.device,
                                         dtype=torch.float)
                loss += cycle_num * penalty * 100
                # if (cycle_num == 0).sum() < 1:
                if cycle_num.min() > 0:
                    self.cyclic_count = min(self.cyclic_count + 1, 30)
                elif cycle_num.median() == 0:
                    self.cyclic_count = max(self.cyclic_count - 1, 0)

                logging.info(f'cycles: {cycle_num.min()}')
                if cycle_info:
                    for i, (class_cycle_path,
                            node_cycle_path) in enumerate(cycles[best_batch]):
                        logging.info(
                            f'cycles_path_info: cycle{i+1}:{class_cycle_path}')
            loss = loss.min()
            if verbose:
                # logging.info(
                print(f'selected {self.node_to_id(enodes[best_batch].bool())}')
        return loss

    def forward(self, embedding, hard=False):
        if self.training:
            if hard:
                return self.hard_sample(embedding).float()
            else:
                # return self.sample(embedding)
                return self.sample_v2(embedding)
        else:
            return self.inference_sample(embedding).float()
