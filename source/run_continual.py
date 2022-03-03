import autograd.numpy as np
import argparse
import json

from json_parser import parse


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'continual'
    solver['algorithm'] = args.algorithm
    spec['solver'] = solver


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str, default='continual',
                        help='the chosen algorithm')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    solver.solve(model, assertion)


if __name__ == '__main__':
    main()
