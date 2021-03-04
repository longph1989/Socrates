import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from autograd import grad
from utils import *

import time

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


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--size', type=str, default='(2,2)',
                        help='the size of the backdoor')
    parser.add_argument('--threshold', type=str, default='0.1',
                        help='the threshold')
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    for target in range(10):
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


if __name__ == '__main__':
    main()
