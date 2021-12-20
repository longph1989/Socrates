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
    assertion['exp_rate'] = args.exp_rate

    assertion['known_stamp'] = args.known_stamp
    assertion['stamp_pos'] = args.stamp_pos
    assertion['stamp_size'] = args.stamp_size

    assertion['total_imgs'] = args.total_imgs
    assertion['num_imgs'] = args.num_imgs
    assertion['num_repair'] = args.num_repair
    
    assertion['clean_atk'] = args.clean_atk
    assertion['clean_acc'] = args.clean_acc
    assertion['time_limit'] = args.time_limit

    if 'mnist' in args.dataset:
        assertion['dataset'] = 'mnist'
    elif 'cifar' in args.dataset:
        assertion['dataset'] = 'cifar'

    assertion['pathX'] = args.pathX
    assertion['pathY'] = args.pathY

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'backdoor_repair'
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


def run_cleansing(args):
    print('Backdoor target = {} with total imgs = {}, clean atk = {}, clean acc = {}, and num repair = {}'.
        format(args.target, args.total_imgs, args.clean_atk, args.clean_acc, args.num_repair))

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    args.pathX, args.pathY = get_dataset(args.dataset)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    solver.solve(model, assertion)


def main():
    start = time.time()

    np.set_printoptions(threshold=20)

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--exp_rate', type=float, default=0.80,
                        help='the expected success rate of the trigger before cleansing')
    parser.add_argument('--clean_atk', type=float, default=0.10,
                        help='the success rate of the same trigger after cleansing')
    parser.add_argument('--clean_acc', type=float, default=0.90,
                        help='the accuracy of the clean model')
    parser.add_argument('--time_limit', type=float, default=60.0,
                        help='the time limit of the optimizer')
    parser.add_argument('--target', type=int,
                        help='the target used in verify and attack')

    parser.add_argument('--known_stamp', action='store_true',
                        help='is the stamp known or need to be generated')
    parser.add_argument('--stamp_pos', type=int, default=0,
                        help='the position of the stamp')
    parser.add_argument('--stamp_size', type=int, default=3,
                        help='the size of the stamp')
    
    parser.add_argument('--algorithm', type=str, default='backdoor_repair',
                        help='the chosen algorithm')
    parser.add_argument('--total_imgs', type=int, default=10000,
                        help='the total number of images')
    parser.add_argument('--num_imgs', type=int, default=10,
                        help='the number of images used to repair each time')
    parser.add_argument('--num_repair', type=int, default=10,
                        help='the number of repair')
    parser.add_argument('--dataset', type=str,
                        help='the data set for BACKDOOR experiments')

    args = parser.parse_args()

    run_cleansing(args)

    end = time.time()

    t = round(end - start)
    m = int(t / 60)
    s = t - 60 * m

    print('\nRunning time = {}m {}s'.format(m, s))


if __name__ == '__main__':
    main()
