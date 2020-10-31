import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *

from solver.lib_solvers import DeepCegar


import time
import gc



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
    elif args.algorithm == 'deepcegar':
        solver['max_ref'] = '5'

    spec['solver'] = solver


def update_bounds(args, model, x0, lower, upper):
    eps = np.full(x0.shape, args.eps)

    if args.dataset == 'cifar_conv':
        eps[0:1024] = eps[0:1024] / 0.2023
        eps[1024:2048] = eps[1024:2048] / 0.1994
        eps[2048:3072] = eps[2048:3072] / 0.2010
    elif args.dataset == 'mnist_conv':
        eps = eps / 0.3081

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
    parser.add_argument('--eps', type=float,
                        help='the distance value')
    parser.add_argument('--dataset', type=str,
                        help='the data set for ERAN experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    lower = model.lower
    upper = model.upper

    if args.dataset == 'cifar_conv':
        pathX = 'benchmark/eran/data/cifar_conv/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif args.dataset == 'cifar_fc':
        pathX = 'benchmark/eran/data/cifar_fc/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif args.dataset == 'mnist_conv':
        pathX = 'benchmark/eran/data/mnist_conv/'
        pathY = 'benchmark/eran/data/labels/y_mnist.txt'
    elif args.dataset == 'mnist_fc':
        pathX = 'benchmark/eran/data/mnist_fc/'
        pathY = 'benchmark/eran/data/labels/y_mnist.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    for i in range(10):    
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))

        output_x0 = model.apply(x0)
        lbl_x0 = np.argmax(output_x0, axis=1)[0]

        print('Data {}\n'.format(i))
        print('x0 = {}'.format(x0))
        print('output_x0 = {}'.format(output_x0))
        print('lbl_x0 = {}'.format(lbl_x0))
        print('y0 = {}\n'.format(y0s[i]))

        t0 = time.time()

        if lbl_x0 == y0s[i]:
            update_bounds(args, model, x0, lower, upper)
            print('Run at data {}\n'.format(i))
            solver.solve(model, assertion)
        else:
            print('Skip at data {}'.format(i))

        t1 = time.time()

        print('time = {}'.format(t1 - t0))
        print('\n============================\n')


if __name__ == '__main__':
    main()
