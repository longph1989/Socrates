import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *


import sys
sys.path.append("/Users/bing.sun/workspace/Fairness/Socrates/Socrates_git")


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
        pathX = '../benchmark/fairness/bank/data/'
        pathY = '../benchmark/fairness/bank/data/labels.txt'
    elif args.dataset == 'census':
        pathX = '../benchmark/fairness/census/data/'
        pathY = '../benchmark/fairness/census/data/labels.txt'
    elif args.dataset == 'credit':
        pathX = '../benchmark/fairness/credit/data/'
        pathY = '../benchmark/fairness/credit/data/labels.txt'
    elif args.dataset == 'FairSquare':
        pathX = '../benchmark/fairness/FairSquare/data/'
        pathY = '../benchmark/fairness/FairSquare/data/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    assertion['x0'] = pathX + 'data' + str(0) + '.txt'

    solver.solve(model, assertion)
    print('\n============================\n')
    '''

    #test accuracy
    ori_acc = 0
    fixed_acc = 0
    for i in range(32000):
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))
        x0_ = x0.copy()
        x0_[4] = 0
        #x0_[10] = 0
        #x0_[2] = 0


        y = np.argmax(model.apply(x0), axis=1)[0]

        #y_, layer_op = model.apply_intermediate(x0)
        #y_ = np.argmax(y_, axis=1)[0]
        y_ = np.argmax(model.apply(x0_), axis=1)[0]

        if y != y0s[i]:
            ori_acc = ori_acc + 1

        if y_ != y0s[i]:
            fixed_acc = fixed_acc + 1


    print("Accuracy of ori network: %fs\n" % ((32000 - ori_acc) / 320))
    print("Accuracy of fxied network: %fs\n" % ((32000 - fixed_acc) / 320))

    print('\n============================\n')
    '''
if __name__ == '__main__':
    main()
