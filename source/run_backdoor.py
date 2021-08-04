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

    assertion['rate'] = args.rate
    assertion['threshold'] = args.threshold

    assertion['alpha'] = args.alpha
    assertion['beta'] = args.beta

    assertion['atk_only'] = args.atk_only

    if args.atk_only:
        assertion['atk_pos'] = args.atk_pos

    assertion['total_imgs'] = args.total_imgs
    assertion['num_imgs'] = args.num_imgs

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


def get_dataset(dataset):
    if dataset == 'cifar_conv':
        pathX = 'benchmark/eran/data/cifar_conv/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif dataset == 'cifar_fc':
        pathX = 'benchmark/eran/data/cifar_fc/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif dataset == 'mnist_conv':
        pathX = 'benchmark/eran/data/mnist_conv_full/'
        pathY = 'benchmark/eran/data/labels/y_mnist_full.txt'
    elif dataset == 'mnist_fc':
        pathX = 'benchmark/eran/data/mnist_fc_full/'
        pathY = 'benchmark/eran/data/labels/y_mnist_full.txt'

    return pathX, pathY


def run_attack(args):
    print('Backdoor target = {} with size = {}, total imgs = {}, num imgs = {}, and attack only = {} at position = {}'.
        format(args.target, args.size, args.total_imgs, args.num_imgs, args.atk_only, args.atk_pos))

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    args.pathX, args.pathY = get_dataset(args.dataset)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    res, stamp = solver.solve(model, assertion)


def run_verify(zipped_args):
    start, end, args = zipped_args[0], zipped_args[1], zipped_args[2]

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    args.pathX, args.pathY = get_dataset(args.dataset)

    bd_target, fa_target, sb_target = [], [], []

    for target in range(start, end):
        args.target = target

        print('Backdoor target = {} with size = {}, total imgs = {}, num imgs = {}, rate = {}, alpha = {}, beta = {}, and threshold = {}'.
            format(args.target, args.size, args.total_imgs, args.num_imgs, args.rate, args.alpha, args.beta, args.threshold))

        add_assertion(args, spec)
        add_solver(args, spec)

        model, assertion, solver, display = parse(spec)

        res, sbi = solver.solve(model, assertion)

        if res is not None:
            bd_target.append(res)
        
            if sbi is None:
                fa_target.append(res)
            else:
                sb_target.append(res)

    return bd_target, fa_target, sb_target


def run_verify_parallel(args):
    bd_target_lst, fa_target_lst, sb_target_lst = [], [], []

    output_size = 10
    num_cores = os.cpu_count()

    if args.num_procs > 0:
        pool_size = args.num_procs
    else:
        pool_size = num_cores if num_cores <= output_size else output_size

    quo = int(output_size / pool_size)
    rem = int(output_size % pool_size)

    idx, srt_lst, end_lst, args_lst = 0, [], [], []

    for i in range(pool_size):
        srt_lst.append(idx)
        idx += quo
        if rem > 0:
            idx += 1
            rem -= 1
        end_lst.append(idx)
        args_lst.append(args)

    zipped_args = zip(srt_lst, end_lst, args_lst)

    pool = multiprocessing.Pool(pool_size)

    for bd_target, fa_target, sb_target in pool.map(run_verify, zipped_args):
        bd_target_lst += bd_target
        fa_target_lst += fa_target
        sb_target_lst += sb_target
    pool.close()

    print('\n==============\n')

    if len(bd_target_lst) == 0:
        print('No backdoor')
    else:
        bd_target_lst.sort()
        for target in bd_target_lst:
            print('Detect backdoor with target = {}'.format(target))

    if len(fa_target_lst) == 0:
        print('No false alarm')
    else:
        fa_target_lst.sort()
        for target in fa_target_lst:
            print('False alarm with target = {}'.format(target))

    if len(sb_target_lst) == 0:
        print('No stamp')
    else:
        sb_target_lst.sort()
        for target in sb_target_lst:
            print('Stamp with target = {}'.format(target))


def main():
    start = time.time()

    np.set_printoptions(threshold=20)

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--size', type=int, default=3,
                        help='the size of the backdoor')
    parser.add_argument('--rate', type=float, default=0.90,
                        help='the success rate')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='the threshold')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='the alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='the beta')
    parser.add_argument('--target', type=int,
                        help='the target used in verify and attack')

    # for attacking
    parser.add_argument('--atk_only', action='store_true',
                        help='turn on/off attack')
    parser.add_argument('--atk_pos', type=int,
                        help='the attack position')
    
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')
    parser.add_argument('--num_procs', type=int, default=0,
                        help='the number of processes')
    parser.add_argument('--total_imgs', type=int, default=10000,
                        help='the number of images')
    parser.add_argument('--num_imgs', type=int, default=100,
                        help='the number of images')
    parser.add_argument('--dataset', type=str,
                        help='the data set for BACKDOOR experiments')

    args = parser.parse_args()

    if args.atk_only:
        run_attack(args)
    else:
        run_verify_parallel(args)

    end = time.time()

    t = round(end - start)
    m = int(t / 60)
    s = t - 60 * m

    print('Running time = {}m {}s'.format(m, s))


if __name__ == '__main__':
    main()
