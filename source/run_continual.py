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

    base = 'benchmark/reluplex/specs/prop1/prop1_nnet_'
    models = []

    for i in range(1,6):
        sub_models = []
        for j in range(1,10):
            args.spec = base + str(i) + '_' + str(j) + '.json'

            with open(args.spec, 'r') as f:
                spec = json.load(f)

            add_solver(args, spec)

            model, assertion, solver, display = parse(spec)
            sub_models.append(model)
        models.append(sub_models)
    
    solver.solve(models, assertion)


if __name__ == '__main__':
    main()
