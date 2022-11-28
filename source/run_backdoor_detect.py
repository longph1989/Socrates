import autograd.numpy as np
import argparse
import json

from json_parser import parse
from solver.lib_solvers import *


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'backdoor_detect'
    solver['algorithm'] = args.algorithm
    spec['solver'] = solver


def add_dataset(args, spec):
    assertion = dict()

    assertion['dataset'] = args.dataset
    spec['assert'] = assertion


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--dataset', type=str, default='none',
                        help='the chosen algorithm')
    parser.add_argument('--algorithm', type=str, default='backdoor_detect',
                        help='the chosen algorithm')

    args = parser.parse_args()
    spec = dict()
        
    add_solver(args, spec)
    add_dataset(args, spec)

    model, assertion, solver, display = parse(spec)
    solver.solve(model, assertion)


if __name__ == '__main__':
    main()
