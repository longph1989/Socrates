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
    assertion['eps'] = str(args.eps)

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    solver['algorithm'] = args.algorithm
    if args.algorithm == 'sprt':
        solver['threshold'] = str(args.threshold)
        solver['alpha'] = '0.05'
        solver['beta'] = '0.05'
        solver['delta'] = '0.005'

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
                        help='the data set for rnn experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    lower = model.lower[0]
    upper = model.upper[0]

    if args.dataset == 'jigsaw':
        pathX = 'benchmark/rnn/data/jigsaw/'
        pathY = 'benchmark/rnn/data/jigsaw/labels.txt'
    elif args.dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    for i in range(100):
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))

        shape_x0 = (int(x0.size / 50), 50)

        model.shape = shape_x0
        model.lower = np.full(x0.size, lower)
        model.upper = np.full(x0.size, upper)

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
