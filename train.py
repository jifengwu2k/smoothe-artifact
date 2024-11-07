import logging
import random
import argparse
import torch
from collections import defaultdict
from tqdm import tqdm
import os
import time
from datetime import datetime
import json
import pickle
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
# from pytorch_memlab import MemReporter
from torch.autograd import profiler

from sparse_egraph import SparseEGraph


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def get_args(default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default='examples/cunxi_test_egraph2.dot')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--random_seed', type=int, default=44)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_attributes', type=int, default=5)
    parser.add_argument('--gumbel_tau', type=float, default=1)
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--regularizer", type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--hard', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--depth', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument(
        '--assumption',
        type=str,
        default='independent',
        choices=['independent', 'correlated', 'hybrid', 'neg_correlated'])
    parser.add_argument('--load_cost', action='store_true', default=False)
    parser.add_argument('--acyclic', action='store_true', default=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument("--greedy_ini", action="store_true", default=False)
    parser.add_argument("--cycle_info", action="store_true", default=False)
    parser.add_argument('--quad_cost',
                        type=str,
                        default=None,
                        help='path to the quadratic cost file')
    parser.add_argument('--mlp_cost',
                        type=str,
                        default=None,
                        help='path to the mlp cost file')
    if default:
        return parser.parse_args([])
    else:
        return parser.parse_args()


def sample(egraph, verbose=False, cycle_info=False):
    egraph.eval()
    start_time = time.time()

    # hacky solution for nn.DataParallel
    if hasattr(egraph, 'module'):
        cur_egraph = egraph.module
    else:
        cur_egraph = egraph

    with torch.no_grad():
        enodes = cur_egraph(cur_egraph.embedding, hard=True)
        loss = cur_egraph.compute_loss(enodes,
                                       verbose=verbose,
                                       cycle_info=cycle_info)
        logging.info(f'Sample loss: {loss:.4f}')
        logging.info(f'sampling time: {time.time() - start_time:.4f}')
    cur_egraph.step()
    egraph.train()
    return loss


class EarlyStopper:

    def __init__(self, patience=3, min_delta=0.1):
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.min_delta = min_delta

    def __call__(self, loss):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


def run(args):
    start_time = time.time()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.input_file.endswith('.dot'):
        args.load_cost = True

    set_random_seed(seed_value=args.random_seed)

    if args.gpus > 0:
        device = 'cuda'
    else:
        device = 'cpu'

    egraph = SparseEGraph(args.input_file,
                          hidden_dim=args.hidden_dim,
                          batch_size=args.batch_size,
                          num_attributes=args.num_attributes,
                          gumbel_tau=args.gumbel_tau,
                          soft=True,
                          device=device,
                          load_cost=args.load_cost,
                          gpus=args.gpus,
                          filter_cycles=args.acyclic,
                          eps=args.eps,
                          greedy_ini=args.greedy_ini,
                          assumtion=args.assumption)
    if args.gpus >= 1:
        egraph = egraph.cuda()
    egraph.set_temperature_schedule(args.num_steps)
    egraph.reg = args.regularizer
    if args.quad_cost:
        egraph.init_quad_cost(args.quad_cost)
    if args.mlp_cost:
        egraph.init_mlp_cost(args.mlp_cost)

    lr = args.base_lr
    params_to_optimize = [
        param for param in egraph.parameters() if param.requires_grad
    ]
    if args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(params_to_optimize,
                                    nesterov=True,
                                    momentum=0.9,
                                    lr=lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize,
                                    lr=lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize,
                                     lr=lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'sparse_adam':
        optimizer = torch.optim.SparseAdam(params_to_optimize, lr=lr)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params_to_optimize,
                                        lr=lr,
                                        weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # warmup_step = args.num_steps // 10
    # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda step: (step + 1) / warmup_step
    #     if step < warmup_step else 1)
    # cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.num_steps)
    # scheduler = warmup_scheduler
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, [warmup_scheduler, cos_scheduler],
    #     milestones=[args.num_steps // 2])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps)

    training_log = defaultdict(list)
    logging.info(f'cost per node {egraph.cost_per_node}')
    early_stop = EarlyStopper(patience=args.patience)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    if args.gpus > 1:
        egraph = torch.nn.DataParallel(egraph)
        cur_egraph = egraph.module
    else:
        cur_egraph = egraph
    if args.compile:
        egraph = torch.compile(egraph)

    if args.verbose:
        for_loop = tqdm(range(args.num_steps))
    else:
        for_loop = range(args.num_steps)
    probs = []
    for step in for_loop:
        inf_loss = sample(egraph, cycle_info=args.cycle_info)
        training_log['sample_time'].append(time.time() - start_time)
        # if step == args.num_steps // 2:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.01
        # hard = True if step > args.num_steps // 2 else False
        # optim_goal = 'depth' if args.depth else 'sum'

        enodes, cyclic_loss = egraph(cur_egraph.embedding, hard=False)
        training_log['forward_time'].append(time.time() - start_time)
        loss = cur_egraph.compute_loss(enodes, cyclic_loss)
        probs.append(egraph.probs_class2node)
        loss.backward()
        # max_emb_grad = torch.max(torch.abs(cur_egraph.embedding.grad))
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

        if not args.random:
            # a hacky way to implement random sampling with parameter freezing
            cur_egraph.step()
        training_log['inference_loss'].append(inf_loss.item())
        training_log['loss'].append(loss.item())
        training_log['time'].append(time.time() - start_time)
        if time.time() - start_time > args.time_limit:
            logging.info('time limit reached')
            break
        if step > 1 and inf_loss < 1e4:
            if early_stop(inf_loss):
                break

        # if step == args.num_steps // 2:
        #     egraph.select_batch(new_batch_size=4, enodes=enodes)
        #     egraph.filter_cycles = True
    torch.save(probs, 'probs.pt')

    logging.info(f'finished optimization, now sampling')
    loss = sample(egraph, verbose=args.verbose, cycle_info=True)

    training_log['time'].append(time.time() - start_time)
    training_log['loss'].append(loss.item())

    logging.info(f'training log: {training_log}')
    file_name = os.path.splitext(os.path.basename(
        args.input_file))[0] + '_smoothe'
    file_name += '_depth' if args.depth else ''
    json.dump(training_log, open(f'logs/smoothe_log/{file_name}.json', 'w'))
    logging.info('logs dumped to ' + f'logs/smoothe_log/{file_name}.json')
    logging.info(
        f'best inference loss = {np.min(training_log["inference_loss"])}')
    logging.info(
        f'best inference time = {training_log["time"][np.argmin(training_log["inference_loss"])]}'
    )
    return training_log


if __name__ == '__main__':
    args = get_args()
    run(args)
