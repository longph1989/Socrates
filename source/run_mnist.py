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
    assertion['eps'] = '1e9' # bounds are updated so eps is not necessary

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


def update_bounds(args, model, x0, lower, upper):
    eps = np.full(x0.shape, 0.3)

    model.lower = np.maximum(lower, x0 - eps)
    model.upper = np.minimum(upper, x0 + eps)


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str,
                        help='the chosen algorithm')
    parser.add_argument('--threshold', type=float,
                        help='the threshold in sprt')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    lower = model.lower
    upper = model.upper

    for i in range(50):
        pathX = 'benchmark/mnist_challenge/x_y/x' + str(i) + '.txt'
        pathY = 'benchmark/mnist_challenge/x_y/y' + str(i) + '.txt'

        x0s = np.array(ast.literal_eval(read(pathX)))
        y0s = np.array(ast.literal_eval(read(pathY)))

        for j in range(200):
            x0 = x0s[j]
            assertion['x0'] = str(x0.tolist())

            output_x0 = model.apply(x0)
            lbl_x0 = np.argmax(output_x0, axis=1)[0]

            print('Data {}\n'.format(i * 200 + j))
            print('x0 = {}'.format(x0))
            print('output_x0 = {}'.format(output_x0))
            print('lbl_x0 = {}'.format(lbl_x0))
            print('y0 = {}\n'.format(y0s[j]))

            if lbl_x0 == y0s[j]:
                update_bounds(args, model, x0, lower, upper)
                print('Run at data {}\n'.format(i * 200 + j))
                solver.solve(model, assertion)
            else:
                print('Skip at data {}'.format(i * 200 + j))

            print('\n============================\n')


if __name__ == '__main__':
    main()
