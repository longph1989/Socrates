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
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    for target in range(start, end):
        args.target = str(target)
        args.pathX = 'benchmark/backdoor/data/mnist_fc/'
        args.pathY = 'benchmark/backdoor/data/labels/y_mnist.txt'

        print('\n============================\n')

        print('Backdoor target = {}'.format(target))

        add_assertion(args, spec)
        add_solver(args, spec)

        model, assertion, solver, display = parse(spec)

        solver.solve(model, assertion)

        print('\n============================\n')


def main():
    np.set_printoptions(threshold=20)

    # run((0, 10))

    output_size = 10
    num_cores = os.cpu_count()

    pool_size = num_cores if num_cores <= output_size else output_size

    quo = int(output_size / num_cores)
    rem = int(output_size % num_cores)

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

    pool.map(run, indexes)
    pool.close()


if __name__ == '__main__':
    main()
