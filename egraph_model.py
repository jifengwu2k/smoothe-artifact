import re
import os
import json
import logging
from collections import defaultdict
from egraph_data import EGraphData
from dag_greedy import greedy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, t


class BatchedLinear(nn.Module):

    def __init__(self, batch_size, in_features, out_features):
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.rand(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.rand(batch_size, out_features))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias


class BaseEGraph(nn.Module, EGraphData):

    def __init__(self,
                 input_file,
                 hidden_dim=32,
                 num_attributes=5,
                 gumbel_tau=1.0,
                 dropout=0.0,
                 eps=0.5,
                 batch_size=None,
                 soft=False,
                 embedding_type='lookup',
                 aggregate_type='mean',
                 device='cuda',
                 load_cost=False,
                 greedy_ini=False,
                 compress=False,
                 drop_self_loops=False,
                 share_proj=True):
        nn.Module.__init__(self)
        EGraphData.__init__(
            self,
            input_file,
            hidden_dim,
            load_cost,
            compress,
            drop_self_loops,
            device,
        )
        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.gumbel_tau = gumbel_tau
        self.dropout = dropout
        self.eps = eps
        self.embedding_type = embedding_type
        self.aggregate_type = aggregate_type
        self.device = device
        self.minimal_cost = None
        self.soft = soft
        self.p_number = np.e
        self.quadratic_cost = None
        self.greedy_ini = greedy_ini
        self.input_file = input_file
        self.share_proj = share_proj

        if batch_size is None:
            # self.batch_size = 100_000_000 / len(self.eclasses)**2
            # print(f'eclass size: {len(self.eclasses)}')
            num_class = len(self.eclasses)
            if num_class > 5000:
                batch_size = 40000 / num_class
            else:
                batch_size = 80000 / num_class
            self.batch_size = int(np.clip(2**int(np.log2(batch_size)), 1, 512))
            # print(f'Auto set batch size to {self.batch_size}')

        self.set_to_matrix()
        self.init_embedding()
        self.init_params()
        torch.cuda.empty_cache()

    def set_adj(self):
        # [N, M]
        # Enode adjacency matrix points from enode to eclass
        self.node_adj = torch.zeros((len(self.enodes), len(self.eclasses)),
                                    dtype=torch.bool,
                                    device=self.device)

        # [M, N]
        # eclass adjacency matrix contains the enodes in the eclass
        self.class_adj = torch.zeros((len(self.eclasses), len(self.enodes)),
                                     dtype=torch.bool,
                                     device=self.device)

        for enode_id in self.enodes:
            eclass_id = [i for i in self.enodes[enode_id].eclass_id]
            self.node_adj[enode_id, eclass_id] = 1

        for eclass_id in self.eclasses:
            self.class_adj[eclass_id, self.eclasses[eclass_id].enode_id] = 1

    def set_to_matrix(self):
        # set the class2node and node2class in a sparse COO format
        n2c_row_index = []
        n2c_col_index = []
        for enode_id in self.enodes:
            eclass_id = [i for i in self.enodes[enode_id].eclass_id]
            n2c_row_index += [enode_id] * len(eclass_id)
            n2c_col_index += eclass_id
        n2c_row_index = torch.tensor(n2c_row_index).to(self.device)
        n2c_col_index = torch.tensor(n2c_col_index).to(self.device)
        self.node2class = SparseTensor(row=n2c_row_index,
                                       col=n2c_col_index,
                                       value=torch.ones(len(n2c_row_index),
                                                        device=self.device),
                                       sparse_sizes=(len(self.enodes),
                                                     len(self.eclasses)))

        c2n_row_index = []
        c2n_col_index = []
        for eclass_id in self.eclasses:
            enode_id = self.eclasses[eclass_id].enode_id
            c2n_row_index += [eclass_id] * len(enode_id)
            c2n_col_index += enode_id
        c2n_row_index = torch.tensor(c2n_row_index).to(self.device)
        c2n_col_index = torch.tensor(c2n_col_index).to(self.device)
        self.class2node = SparseTensor(row=c2n_row_index,
                                       col=c2n_col_index,
                                       value=torch.ones(len(c2n_row_index),
                                                        device=self.device),
                                       sparse_sizes=(len(self.eclasses),
                                                     len(self.enodes)))

    def init_embedding(self):
        self.embedding = torch.rand(self.batch_size,
                                    len(self.enodes),
                                    self.hidden_dim,
                                    device=self.device)
        self.embedding = self.embedding / np.sqrt(self.hidden_dim)
        if self.greedy_ini:
            self.bias = torch.zeros(len(self.enodes)).to(self.device)
            greedy_idx_list = greedy(self, method="faster", ini_greedy=True)
            self.bias[greedy_idx_list] += self.eps
        self.embedding = nn.Parameter(self.embedding, requires_grad=True)

    def init_params(self):
        if self.embedding_type == 'projection':
            self.emb_layer = torch.nn.Linear(in_features=self.num_attributes,
                                             out_features=self.hidden_dim)
        elif self.embedding_type == 'lookup':
            self.emb_layer = torch.nn.Embedding(num_embeddings=len(
                self.enodes),
                                                embedding_dim=self.hidden_dim)
        if self.share_proj:
            self.enode_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                              out_features=self.hidden_dim)
            self.output_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                               out_features=1)
        else:
            self.node_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                           self.hidden_dim)
            self.output_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                             1)
        self.activation = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.set_attr(self.num_attributes)

    def set_temperature_schedule(self, steps, schedule='constant'):
        if schedule == 'constant':
            self.temperature_schedule = np.ones(steps)
        if schedule == 'linear':
            self.temperature_schedule = np.linspace(1, 1e-3, steps)
        elif schedule == 'log':
            self.temperature_schedule = np.logspace(0, -2, steps)
        self.step_count = 0

    def dense(self, enode_embedding, context_embedding):
        # enode_embedding: [n, hidden_dim]
        # context_embedding: [1, hidden_dim]
        # return: [n, hidden_dim]
        return self.enode_proj(enode_embedding) + self.context_proj(
            context_embedding)

    def projection(self, enode_embedding):
        # enode_embedding: [n, hidden_dim]
        # return: [n]
        return self.output_proj(enode_embedding).squeeze(-1)

    def forward_embedding(self, embedding):
        logit = self.activation(self.enode_proj(embedding))
        logit = self.projection(self.dropout(logit))
        if self.greedy_ini and not self.training and self.step_count == 0:
            logit[0] = logit[0] + self.bias
        return logit

    def set_attr(self, num_attributes):
        if self.embedding_type == 'projection':
            self.enode_attr = torch.randn(len(self.enodes),
                                          num_attributes).to(self.device)
        elif self.embedding_type == 'lookup':
            self.enode_attr = torch.arange(len(self.enodes)).to(self.device)


class AdhocEGraph(BaseEGraph):

    def step(self):
        for eclass in self.eclasses.values():
            eclass.visited_in_nodes = defaultdict(int)
            eclass.included = False
            eclass.predecessor_embeddings = []
            eclass.predecessor_enodes = []
            eclass.normalized_prob = 1
        for enode in self.enodes.values():
            enode.normalized_prob = 0
        self.gumbel_tau = self.temperature_schedule[self.step_count]
        self.step_count += 1

    def get_enode_embedding(self, enode_ids):
        # Sum the embedded attribute over the categories
        # TODO: Consider changing simple sum to projection
        if self.embedding_type == 'projection':
            return self.emb_layer(self.enode_attr[enode_ids])
        elif self.embedding_type == 'lookup':
            return self.emb_layer(self.enode_attr[enode_ids])

    def aggregate_eclass_context(self, eclass):
        if eclass.predecessor_embeddings == []:
            return self.start_embedding
        if self.aggregate_type == 'sum':
            return torch.stack(
                eclass.predecessor_embeddings).sum(dim=0).reshape(1, -1)
        elif self.aggregate_type == 'mean':
            return torch.stack(
                eclass.predecessor_embeddings).mean(dim=0).reshape(1, -1)
        elif self.aggregate_type == 'max':
            return torch.stack(eclass.predecessor_embeddings).max(
                dim=0).values.reshape(1, -1)
        elif self.aggregate_type == 'none':
            return self.start_embedding
        else:
            raise NotImplementedError

    def sample(self):
        if self.soft:
            return self.soft_sample()
        else:
            return self.hard_sample()

    # TODO: make the sampling process batched
    def hard_sample(self):
        # implement depth optimization for hard
        selected = {}
        to_visit = []
        visited = set()

        # initialize to_visit with all source eclasses
        for eclass_id in self.eclasses:
            if len(self.eclasses[eclass_id].in_nodes) == 0:
                to_visit.append(eclass_id)
                self.eclasses[eclass_id].included = True

        while to_visit:
            included = [
                eclass for eclass in self.eclasses
                if self.eclasses[eclass].included
            ]
            # logging.info(f'To visit: {to_visit}')
            # logging.info(f'Included: {included}')
            eclass_id = to_visit.pop()
            if eclass_id in visited:
                continue

            eclass = self.eclasses[eclass_id]
            # TODO: wrap the following in a function
            enode_embedding = self.get_enode_embedding(eclass.enode_id)
            context_embedding = self.aggregate_eclass_context(eclass)
            # use context embedding to update enode embedding
            updated_embedding = self.dropout(
                self.activation(self.dense(enode_embedding,
                                           context_embedding)))
            logits = self.projection(updated_embedding)
            logging.info(f'visiting eclass {eclass_id} with logits {logits}')
            if self.training:
                choice = F.gumbel_softmax(logits,
                                          tau=self.gumbel_tau,
                                          hard=True)
            else:
                choice = torch.eq(logits, logits.max()).float()
            selected[eclass_id] = choice
            sg_choice = eclass.enode_id[choice.argmax()]

            visited.add(eclass_id)
            for out_eclass in self.enodes[sg_choice].eclass_id:
                self.eclasses[out_eclass].included = True

                self.eclasses[out_eclass].normalized_prob *= choice.max()

                # propagate with gumbel softmax signal
                self.eclasses[out_eclass].predecessor_embeddings.append(
                    choice.matmul(updated_embedding))

                # propagate without gumbel softmax signal
                # self.eclasses[out_eclass].predecessor_embeddings.append(
                #     updated_embedding[choice.argmax()])

            for enode in eclass.enode_id:
                for out_eclass in self.enodes[enode].eclass_id:
                    self.eclasses[out_eclass].add_visited_in_node(enode)
                    # If all in_nodes are visited, add to to_visit
                    if self.eclasses[out_eclass].in_nodes == self.eclasses[
                            out_eclass].visited_in_nodes and self.eclasses[
                                out_eclass].included and out_eclass not in visited:
                        to_visit.append(out_eclass)

            if len(to_visit) == 0:
                # Find the an included eclass with the least number of unvisited in_nodes
                min_unvisited_in_nodes = float('inf')
                candidate_eclasses = None
                for eclass_id in self.eclasses:
                    if self.eclasses[eclass_id].included:
                        unvisited_in_nodes = len(
                            self.eclasses[eclass_id].in_nodes) - len(
                                self.eclasses[eclass_id].visited_in_nodes)
                        if unvisited_in_nodes < min_unvisited_in_nodes and unvisited_in_nodes > 0 and eclass_id not in visited:
                            min_unvisited_in_nodes = unvisited_in_nodes
                            candidate_eclasses = eclass_id
                if candidate_eclasses is not None:
                    to_visit.append(candidate_eclasses)

        logging.info(f'Selected: {selected}')
        return selected

    def soft_sample(self):
        selected = {}
        to_visit = []
        visited = set()

        # initialize to_visit with all source eclasses
        for eclass_id in self.eclasses:
            if len(self.eclasses[eclass_id].in_nodes) == 0:
                to_visit.append(eclass_id)
                self.eclasses[eclass_id].included = True

        while to_visit:
            included = [
                eclass for eclass in self.eclasses
                if self.eclasses[eclass].included
            ]
            # logging.info(f'To visit: {to_visit}')
            # logging.info(f'Included: {included}')
            eclass_id = to_visit.pop()
            if eclass_id in visited:
                continue

            eclass = self.eclasses[eclass_id]
            # TODO: wrap the following in a function
            enode_embedding = self.get_enode_embedding(eclass.enode_id)
            context_embedding = self.aggregate_eclass_context(eclass)
            # use context embedding to update enode embedding
            updated_embedding = self.dropout(
                self.activation(self.dense(enode_embedding,
                                           context_embedding)))
            logits = self.projection(updated_embedding)
            logging.info(f'visiting eclass {eclass_id} with logits {logits}')
            if self.training:
                choice = F.gumbel_softmax(logits,
                                          tau=self.gumbel_tau,
                                          hard=False)

                # using softmax instead of gumbel softmax
                # choice = F.softmax(logits, dim=-1)
            else:
                # softmax = F.softmax(logits, dim=-1)
                # print(f'max logits: {softmax.max()}')
                choice = torch.eq(logits, logits.max()).float()
            selected[eclass_id] = choice
            # if isinstance(eclass.normalized_prob, torch.Tensor):
            #     eclass.normalized_prob = 1 - eclass.normalized_prob
            class_depth = 0
            class_prob = 1
            if len(eclass.predecessor_enodes) != 0:
                for enode in eclass.predecessor_enodes:
                    class_prob *= 1 - self.enodes[enode].normalized_prob
                    class_depth += self.enodes[enode].depth
                class_prob = 1 - class_prob
                class_depth /= len(eclass.predecessor_enodes)
            sg_choice = eclass.enode_id[choice.argmax()]

            visited.add(eclass_id)

            for i, enode in enumerate(eclass.enode_id):
                if self.enodes[enode].normalized_prob == 0:
                    self.enodes[enode].normalized_prob = choice[i] * class_prob
                    self.enodes[enode].depth = class_depth + 1
                else:
                    raise ValueError
                for out_eclass in self.enodes[enode].eclass_id:
                    self.eclasses[out_eclass].included = True

                    self.eclasses[out_eclass].predecessor_enodes.append(enode)

                    # propagate with gumbel softmax signal
                    self.eclasses[out_eclass].predecessor_embeddings.append(
                        choice.matmul(updated_embedding))

            for enode in eclass.enode_id:
                for out_eclass in self.enodes[enode].eclass_id:
                    self.eclasses[out_eclass].add_visited_in_node(enode)
                    # If all in_nodes are visited, add to to_visit
                    if self.eclasses[out_eclass].in_nodes == self.eclasses[
                            out_eclass].visited_in_nodes and self.eclasses[
                                out_eclass].included and out_eclass not in visited:
                        to_visit.append(out_eclass)

            if len(to_visit) == 0:
                # Find the an included eclass with the least number of unvisited in_nodes
                min_unvisited_in_nodes = float('inf')
                candidate_eclasses = None
                for eclass_id in self.eclasses:
                    if self.eclasses[eclass_id].included:
                        unvisited_in_nodes = len(
                            self.eclasses[eclass_id].in_nodes) - len(
                                self.eclasses[eclass_id].visited_in_nodes)
                        if unvisited_in_nodes < min_unvisited_in_nodes and unvisited_in_nodes > 0 and eclass_id not in visited:
                            min_unvisited_in_nodes = unvisited_in_nodes
                            candidate_eclasses = eclass_id
                if candidate_eclasses is not None:
                    to_visit.append(candidate_eclasses)

        logging.info(f'Selected: {selected}')
        return selected

    def forward(self, selected, backward=True, optim_goal='sum', debug=False):
        assert not debug
        # use forward to compute loss
        cost = 0
        cost_per_node = self.cost_per_node
        if not self.soft:
            for eclass_id in selected:
                enode_id = self.eclasses[eclass_id].enode_id
                enode_weight = cost_per_node[enode_id]
                chain = self.eclasses[eclass_id].normalized_prob
                assert chain == 1 or chain.item() == 1
                cost += chain * torch.matmul(selected[eclass_id], enode_weight)
        else:
            for enode in self.enodes:
                enode_weight = cost_per_node[enode]
                if optim_goal == 'depth':
                    enode_weight = self.p_number**self.enodes[enode].depth
                cost += enode_weight * self.enodes[enode].normalized_prob
        loss = cost
        if backward:
            loss.backward()
        loss = loss.item()
        return cost
