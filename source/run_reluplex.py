import autograd.numpy as np
import argparse
import json

from json_parser import parse


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

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    solver.solve(model, assertion)


if __name__ == '__main__':
    main()
