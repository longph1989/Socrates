import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse_model, parse_assertion, parse_solver
from autograd import grad
from utils import *

from collections import Counter 

import time

def add_assertion(args, spec):
    assertion = dict()

    assertion['size'] = args.size
    assertion['position'] = args.position
    assertion['target'] = args.target
    assertion['threshold'] = args.threshold

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'backdoor'
    solver['algorithm'] = args.algorithm

    spec['solver'] = solver


def get_position(position_lst, threshold):
    occurence_count = Counter(position_lst)
    position_candidate = occurence_count.most_common(1)

    if position_candidate[0][1] > threshold:
        return position_candidate[0][0]
    else:
        return None


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--size', type=str, default='(2,2)',
                        help='the size of the backdoor')
    parser.add_argument('--position', type=str, default='0',
                        help='the position of the backdoor')
    parser.add_argument('--target', type=str,
                        help='the backdoor target')
    parser.add_argument('--threshold', type=str, default='0.1',
                        help='the threshold')
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')
    parser.add_argument('--dataset', type=str,
                        help='the data set for backdoor experiments')
    parser.add_argument('--num_tests', type=int, default=100,
                        help='maximum number of tests')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    for target in range(10):
        args.target = str(target)

        print('\n============================\n')

        print('Backdoor target = {}'.format(target))

        position_lst, total_cnt = [], 0

        pathX = 'benchmark/backdoor/data/mnist_fc/'
        pathY = 'benchmark/backdoor/data/labels/y_mnist.txt'

        y0s = np.array(ast.literal_eval(read(pathY)))

        model = parse_model(spec['model'])

        for i in range(args.num_tests):
            path = pathX + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(path)))

            output_x0 = model.apply(x0)
            lbl_x0 = np.argmax(output_x0, axis=1)[0]

            if lbl_x0 == y0s[i] and lbl_x0 != target:
                jac = grad(model.apply)
                position_lst.append(np.argmax(jac(x0, target)))
                total_cnt += 1

        args.position = get_position(position_lst, total_cnt / 2)

        print('Backdoor position = {}'.format(args.position))

        print('\n============================\n')

        if args.position is None: continue
        else: args.position = str(args.position)

        add_assertion(args, spec)
        add_solver(args, spec)

        assertion = parse_assertion(spec['assert'])
        solver = parse_solver(spec['solver'])

        backdoor_cnt = 0

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
                if solver.solve(model, assertion):
                    backdoor_cnt += 1
            else:
                print('Skip at Data {}'.format(i))

            print('\n============================\n')

        print('Backdoor count = {}'.format(backdoor_cnt))
        print('Total count = {}'.format(total_cnt))

        print('\n============================\n')


if __name__ == '__main__':
    main()
