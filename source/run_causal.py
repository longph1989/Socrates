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
        pathX = 'benchmark/causal/bank/data/'
        pathY = 'benchmark/causal/bank/data/labels.txt'
    elif args.dataset == 'census':
        pathX = 'benchmark/causal/census/data/'
        pathY = 'benchmark/causal/census/data/labels.txt'
    elif args.dataset == 'credit':
        pathX = 'benchmark/causal/credit/data/'
        pathY = 'benchmark/causal/credit/data/labels.txt'
    elif args.dataset == 'FairSquare':
        pathX = 'benchmark/causal/FairSquare/data/'
        pathY = 'benchmark/causal/FairSquare/data/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    assertion['x0'] = pathX + 'data' + str(0) + '.txt'

    solver.solve(model, assertion)


    print('\n============================\n')

if __name__ == '__main__':
    main()
