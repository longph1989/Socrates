import autograd.numpy as np
import argparse
import json
import ast
import os

from json_parser import parse
from autograd import grad
from utils import *

import time
import multiprocessing

def add_assertion(args, spec):
    assertion = dict()

    assertion['target'] = args.target
    assertion['size'] = args.size
    assertion['threshold'] = args.threshold
    assertion['fix_pos'] = args.fix_pos

    if 'mnist' in args.dataset:
        assertion['dataset'] = 'mnist'
    elif 'cifar' in args.dataset:
        assertion['dataset'] = 'cifar'

    assertion['pathX'] = args.pathX
    assertion['pathY'] = args.pathY

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'backdoor'
    solver['algorithm'] = args.algorithm

    spec['solver'] = solver


def run(indexes):
    start, end = indexes[0], indexes[1]

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--size', type=str, default='(3,3)',
                        help='the size of the backdoor')
    parser.add_argument('--threshold', type=str, default='0.1',
                        help='the threshold')
    parser.add_argument('--fix_pos', type=str, default='True',
                        help='turn on/off fixed position')
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')
    parser.add_argument('--dataset', type=str,
                        help='the data set for BACKDOOR experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    if args.dataset == 'cifar_conv':
        args.pathX = 'benchmark/cegar/data/cifar_conv/'
        args.pathY = 'benchmark/cegar/data/labels/y_cifar.txt'
    elif args.dataset == 'cifar_fc':
        args.pathX = 'benchmark/cegar/data/cifar_fc/'
        args.pathY = 'benchmark/cegar/data/labels/y_cifar.txt'
    elif args.dataset == 'mnist_conv':
        args.pathX = 'benchmark/cegar/data/mnist_conv/'
        args.pathY = 'benchmark/cegar/data/labels/y_mnist.txt'
    elif args.dataset == 'mnist_fc':
        args.pathX = 'benchmark/cegar/data/mnist_fc/'
        args.pathY = 'benchmark/cegar/data/labels/y_mnist.txt'

    bd_target, fa_target = [], []

    for target in range(start, end):
        args.target = str(target)

        # print('\n============================\n')

        add_assertion(args, spec)
        add_solver(args, spec)

        model, assertion, solver, display = parse(spec)

        print('Backdoor target = {} with threshold = {} and fix_pos = {}'.format(args.target, args.threshold, args.fix_pos))

        res, stamp = solver.solve(model, assertion)

        if res is not None:
            bd_target.append(res)
            if stamp is None:
                fa_target.append(res)

        # print('\n============================\n')

    return bd_target, fa_target


def main():
    np.set_printoptions(threshold=20)

    # res = run((0, 10))

    output_size = 10
    num_cores = os.cpu_count()

    pool_size = num_cores if num_cores <= output_size else output_size

    quo = int(output_size / pool_size)
    rem = int(output_size % pool_size)

    idx, start, end = 0, [], []

    for i in range(pool_size):
        start.append(idx)
        idx += quo
        if rem > 0:
            idx += 1
            rem -= 1
        end.append(idx)

    indexes = zip(start, end)

    pool = multiprocessing.Pool(pool_size)
    bd_target_lst, fa_target_lst = [], []

    for bd_target, fa_target in pool.map(run, indexes):
        bd_target_lst += bd_target
        fa_target_lst += fa_target
    pool.close()

    if len(bd_target_lst) == 0:
        print('No backdoor!')
    else:
        bd_target_lst.sort()
        for target in bd_target_lst:
            print('Detect backdoor with target = {}!'.format(target))

    if len(fa_target_lst) == 0:
        print('No false alarm!')
    else:
        fa_target_lst.sort()
        for target in fa_target_lst:
            print('False alarm with target = {}!'.format(target))


if __name__ == '__main__':
    main()
