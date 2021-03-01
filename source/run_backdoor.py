import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *

from solver.lib_solvers import DeepCegar

import time

def add_assertion(args, spec):
    assertion = dict()

    assertion['size'] = '(1,1)'
    assertion['position'] = '783'
    assertion['target'] = args.target

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
    parser.add_argument('--target', type=str,
                        help='the backdoor target')
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')
    parser.add_argument('--dataset', type=str,
                        help='the data set for backdoor experiments')
    parser.add_argument('--num_tests', type=int, default=10,
                        help='maximum number of tests')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    for target in range(10):
        args.target = str(target)

        print('\n============================\n')

        print('Backdoor target = {}'.format(target))

        print('\n============================\n')

        add_assertion(args, spec)
        add_solver(args, spec)

        model, assertion, solver, display = parse(spec)

        pathX = 'benchmark/backdoor/data/mnist_fc/'
        pathY = 'benchmark/backdoor/data/labels/y_mnist.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        for i in range(args.num_tests):
            assertion['x0'] = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(assertion['x0'])))

            output_x0 = model.apply(x0)
            lbl_x0 = np.argmax(output_x0, axis=1)[0]

            print('Data {}\n'.format(i))
            print('x0 = {}'.format(x0))
            print('output_x0 = {}'.format(output_x0))
            print('lbl_x0 = {}'.format(lbl_x0))
            print('y0 = {}\n'.format(y0s[i]))

            if lbl_x0 == y0s[i] and lbl_x0 != target:
                res = solver.solve(model, assertion)
            else:
                print('Skip at Data {}'.format(i))

            print('\n============================\n')


if __name__ == '__main__':
    main()
