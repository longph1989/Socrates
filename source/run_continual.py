import autograd.numpy as np
import argparse
import json

from json_parser import parse
from solver.lib_solvers import *


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'continual'
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
    parser.add_argument('--algorithm', type=str, default='continual',
                        help='the chosen algorithm')

    args = parser.parse_args()
    models = []

    if args.dataset == 'acasxu':
        base = 'benchmark/reluplex/specs/prop1/prop1_nnet_'
        for i in range(1,6):
            sub_models = []
            for j in range(1,10):
                args.spec = base + str(i) + '_' + str(j) + '.json'

                with open(args.spec, 'r') as f:
                    spec = json.load(f)

                add_solver(args, spec)
                add_dataset(args, spec)

                model, assertion, solver, display = parse(spec)
                sub_models.append(model)
            models.append(sub_models)
        solver.solve(models, assertion)
    elif args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'census':
        spec = dict()
        
        add_solver(args, spec)
        add_dataset(args, spec)

        model, assertion, solver, display = parse(spec)
        solver.solve(models, assertion)
    else:
        assert False


if __name__ == '__main__':
    main()
