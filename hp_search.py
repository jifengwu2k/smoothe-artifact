import subprocess as sp
import argparse
import os
import torch
import numpy as np
from train import run
from train import get_args as get_train_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default='examples/gym_data/tensat/cyclic/resnet50.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--acyclic', action='store_true', default=False)
    return parser.parse_args()


def call_command(args):
    try:
        log = run(args)
    except RuntimeError as e:
        # print("Caught CUDA Error: Insufficient resources")
        if args.batch_size is None:
            args.batch_size = 64
        elif args.batch_size > 1:
            args.batch_size = args.batch_size // 2
            # print(f"Halfing batch size to {args.batch_size}")
        else:
            return None
        log = call_command(args)
        args.batch_size = None
    except ValueError as ve:
        # print(f"Caught ValueError: {str(ve)}")
        return None
    except Exception as ex:
        # print(f"Caught an unexpected exception: {str(ex)}")
        return None
    return log


if __name__ == "__main__":
    args = get_args()

    best_hp = None
    best_loss = float('inf')
    best_time = float('inf')

    optimizers = ['rmsprop', 'adamw']
    lrs = [1e-1, 1e-2]
    assumptions = ['independent', 'correlated', 'hybrid']
    regs = [1e-2, 1e-4]
    for optimizer in optimizers:
        for lr in lrs:
            for assumption in assumptions:
                for reg in regs:
                    train_args = get_train_args(default=True)
                    train_args.num_steps = 100
                    train_args.input_file = args.input_file
                    train_args.gpus = args.gpus
                    train_args.batch_size = args.batch_size

                    train_args.optimizer = optimizer
                    train_args.assumption = assumption
                    train_args.base_lr = lr
                    train_args.regularizer = reg
                    train_args.acyclic = args.acyclic

                    print(
                        f'optimizer: {optimizer}, lr: {lr}, assumption: {assumption}, reg: {reg}'
                    )
                    log = call_command(train_args)
                    if log is None:
                        continue
                    min_loss = min(log['inference_loss'])
                    min_iter = np.argmin(log['inference_loss'])
                    time = log['time'][min_iter]
                    print(f'Min loss: {min_loss}, time: {time}')

                    if (min_loss < best_loss) or (min_loss == best_loss
                                                  and time < best_time):
                        best_hp = {
                            'optimizer': optimizer,
                            'lr': lr,
                            'assumption': assumption,
                            'reg': reg
                        }
                        best_loss = min_loss
                        best_time = time
    print(f'Best hyperparameters: {best_hp}, loss: {best_loss}')
