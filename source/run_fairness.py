import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *


def add_assertion(args, spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = '1e9' # eps is not necessary in this experiment

    spec['assert'].update(assertion)


def add_solver(args, spec):
    solver = dict()

    solver['algorithm'] = args.algorithm
    if args.algorithm == 'sprt':
        solver['threshold'] = str(args.threshold)
        solver['alpha'] = '0.05'
        solver['beta'] = '0.05'
        solver['delta'] = '0.005'
    elif args.algorithm == 'deepcegar':
        solver['max_ref'] = '5'

    spec['solver'] = solver


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str,
                        help='the chosen algorithm')
    parser.add_argument('--threshold', type=float,
                        help='the threshold in sprt')
    parser.add_argument('--eps', type=float,
                        help='the distance value')
    parser.add_argument('--dataset', type=str,
                        help='the data set for fairness experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    if args.dataset == 'bank':
        pathX = 'benchmark/fairness/bank/data/'
        pathY = 'benchmark/fairness/bank/data/labels.txt'
    elif args.dataset == 'census':
        pathX = 'benchmark/fairness/census/data/'
        pathY = 'benchmark/fairness/census/data/labels.txt'
    elif args.dataset == 'credit':
        pathX = 'benchmark/fairness/credit/data/'
        pathY = 'benchmark/fairness/credit/data/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    for i in range(1):
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))

        output_x0 = model.apply(x0)
        lbl_x0 = np.argmax(output_x0, axis=1)[0]

        print('Data {}\n'.format(i))
        print('x0 = {}'.format(x0))
        print('output_x0 = {}'.format(output_x0))
        print('lbl_x0 = {}'.format(lbl_x0))
        print('y0 = {}\n'.format(y0s[i]))

        if lbl_x0 == y0s[i]:
            print('Run at data {}\n'.format(i))
            solver.solve(model, assertion)
        else:
            print('Skip at data {}'.format(i))

        print('\n============================\n')


if __name__ == '__main__':
    main()
