import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *

from solver.lib_solvers import DeepCegar

import time

def add_assertion(args, spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = '1e9' # bounds are updated so eps is not necessary

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'deepcegar'
    solver['algorithm'] = args.algorithm

    solver['has_ref'] = str(args.has_ref)
    solver['max_ref'] = str(args.max_ref)
    solver['ref_typ'] = str(args.ref_typ)
    solver['max_sus'] = str(args.max_sus)

    spec['solver'] = solver


def update_bounds(args, model, x0, lower, upper):
    eps = np.full(x0.shape, args.eps)
    model.lower = np.maximum(lower, x0 - eps)
    model.upper = np.minimum(upper, x0 + eps)


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str,
                        help='the chosen algorithm')
    parser.add_argument('--eps', type=float,
                        help='the distance value')
    parser.add_argument('--has_ref', action='store_true',
                        help='turn on/off refinement')
    parser.add_argument('--max_ref', type=int, default=20,
                        help='maximum times of refinement')
    parser.add_argument('--ref_typ', type=int, default=0,
                        help='type of refinement')
    parser.add_argument('--max_sus', type=int, default=1,
                        help='maximum times of finding adversarial sample')
    parser.add_argument('--dataset', type=str,
                        help='the data set for CEGAR experiments')
    parser.add_argument('--num_tests', type=int, default=100,
                        help='maximum number of tests')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    lower = model.lower
    upper = model.upper

    pathX = 'benchmark/cegar/data/mnist_fc/'
    pathY = 'benchmark/cegar/data/labels/y_mnist.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    for i in range(args.num_tests):
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
            best_verified, best_failed = 0, 1e9
            eps, step_eps = 0.01, 0.01

            while True:
                t0 = time.time()

                args.eps = eps
                update_bounds(args, model, x0, lower, upper)
                print('Run at data {}\n'.format(i))

                res = solver.solve(model, assertion)

                if res == 1:
                    print('Verified at {:.3f}'.format(eps))
                    best_verified = max(best_verified, eps)
                elif res == 0:
                    print('Failed at {:.3f}'.format(eps))
                    best_failed = min(best_failed, eps)
                else: break

                t1 = time.time()

                print('time = {}'.format(t1 - t0))
                print('\n============================\n')

                if best_verified == round(best_failed - 0.001, 3): break

                if res == 1:
                    if step_eps == 0.01:
                        eps = round(eps + step_eps, 3)
                    elif step_eps == -0.005:
                        step_eps = 0.001
                        eps = round(eps + step_eps, 3)
                    elif step_eps == 0.001:
                        eps = round(eps + step_eps, 3)
                elif res == 0:
                    if step_eps == 0.01:
                        step_eps = -0.005
                        eps = round(eps + step_eps, 3)
                    elif step_eps == -0.005:
                        step_eps = -0.001
                        eps = round(eps + step_eps, 3)
                    elif step_eps == -0.001:
                        eps = round(eps + step_eps, 3)

            print("Image {} Verified at {:.3f} and Failed at {:.3f}".format(i, best_verified, best_failed))
        else:
            print('Skip at data {}'.format(i))
            print("Image {} Verified at {:.3f} and Failed at {:.3f}".format(i, -1, -1))
            res = -1

        print('\n============================\n')





if __name__ == '__main__':
    main()
