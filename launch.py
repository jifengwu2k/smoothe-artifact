import subprocess as sp
import numpy as np
import argparse
from hp_search import call_command
from train import get_args as get_train_args
import os
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--method',
                        type=str,
                        default='cbc',
                        choices=['cbc', 'scip', 'cplex', 'smoothe'])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--acyclic', action='store_true', default=False)
    parser.add_argument('--greedy_init', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='box')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--cost',
                        type=str,
                        default='linear',
                        choices=['linear', 'quad', 'mlp'])
    return parser.parse_args()


def load_hp(dataset):
    if dataset == 'rover':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'independent',
            'reg': 1e-2
        }
    elif dataset == 'tensat':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'independent',
            'reg': 1e-2
        }
    elif dataset == 'flexc':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'correlated',
            'reg': 1e-4
        }
    elif dataset == 'diospyros':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'independent',
            'reg': 1e-2
        }
    elif dataset == 'impress':
        hp = {
            'optimizer': 'adamw',
            'lr': 1e-2,
            'assumption': 'correlated',
            'reg': 1e-2
        }
    elif dataset == 'set':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'independent',
            'reg': 1e-2
        }
    elif dataset == 'maxsat':
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'independent',
            'reg': 1e-2
        }
    else:
        # default hp setting
        hp = {
            'optimizer': 'rmsprop',
            'lr': 1e-2,
            'assumption': 'hybrid',
            'reg': 1e-2
        }
    return hp


def launch(path, dataset, args, exp_id):
    all_logs = {}
    if args.method == 'smoothe':
        hp = load_hp(dataset)
        train_args = get_train_args(default=True)
        train_args.num_steps = args.steps
        train_args.gpus = args.gpus
        train_args.acyclic = args.acyclic
        train_args.batch_size = args.batch_size
        train_args.greedy_ini = args.greedy_init
        train_args.random_seed = exp_id

        train_args.optimizer = hp['optimizer']
        train_args.assumption = hp['assumption']
        train_args.base_lr = hp['lr']
        train_args.regularizer = hp['reg']

        # print(
        #     f'optimizer: {train_args.optimizer}, lr: {train_args.base_lr}',
        #     f'assumption: {train_args.assumption}, reg: {train_args.regularizer}'
        # )

    for file in os.listdir(path):
        if not file.endswith('.json') and not file.endswith('.dot'):
            continue
        print(f'running on {file}')
        if args.method == 'smoothe':

            if args.cost != 'linear':
                cost_file = os.path.join('nonlinear_cost', path, file)
                cost_file = cost_file.replace('.json',
                                              f'_{args.cost}_cost.pkl')
                cost_file = cost_file.replace('.dot', f'_{args.cost}_cost.pkl')
                if args.cost == 'mlp':
                    train_args.mlp_cost = cost_file
                elif args.cost == 'quad':
                    train_args.quad_cost = cost_file

            train_args.input_file = os.path.join(path, file)
            log = call_command(train_args)

            if log is None:
                min_loss = None
                time = None
            else:
                min_loss = min(log['inference_loss'])
                min_iter = np.argmin(log['inference_loss'])
                time = log['time'][min_iter]
                # print(f'Achieved loss: {min_loss}, time: {time} \n')
            all_logs[file] = log

        elif args.method in ['cplex', 'cbc', 'scip']:
            for time_limit in [60 * 15]:
                command = 'python ilp.py' + f' --time_limit {time_limit}'
                command += f' --input_file {os.path.join(path, file)} --solver {args.method}'
                command += ' --acyclic ' if args.acyclic else ''
                command += ' --verbose '

                # print(f'Running command: {command}')
                with open(f'logs/{dataset}_{file}_{args.method}.log',
                          'w') as f:
                    sp.run(command, shell=True, stdout=f, stdin=f)
        elif args.method == 'oracle':
            time_limit = 3600 * 10
            command = 'python ilp.py' + f' --time_limit {time_limit}'
            command += f' --input_file {os.path.join(path, file)} --solver cplex'
            command += ' --acyclic ' if args.acyclic else ''
            command += ' --verbose '
            with open(f'logs/{dataset}_{file}_cplex_oracle.log', 'w') as f:
                sp.run(command, shell=True, stdout=f, stdin=f)
    if args.method == 'smoothe':
        file_name = f'{dataset}_{args.cost}_{exp_id}'
        if not args.greedy_init:
            file_name += '_v'
        file_path = os.path.join('logs', f'{file_name}.json')
        json.dump(all_logs, open(file_path, 'w'))
        print(f'All logs saved to {file_path}')


if __name__ == "__main__":
    args = get_args()

    for i in range(args.repeat):
        if args.dataset in ['realistic', 'synthetic', 'all']:
            if args.dataset == 'realistic':
                all_dataset = [
                    'rover', 'tensat', 'flexc', 'diospyros', 'impress'
                ]
            elif args.dataset == 'synthetic':
                all_dataset = ['set', 'maxsat']
            elif args.dataset == 'all':
                all_dataset = [
                    'rover', 'tensat', 'flexc', 'diospyros', 'impress', 'set',
                    'maxsat'
                ]
            for dataset in all_dataset:
                path = os.path.join('./dataset/', dataset)
                launch(path, dataset, args, i)

        else:
            path = os.path.join('./dataset/', args.dataset)
            launch(path, args.dataset, args, i)
