import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os
import time
import random
import math

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *
from solver.deepcegar_impl import Poly

import matplotlib.pyplot as plt


class BackDoorImpl():
    def __solve_backdoor(self, model, spec, display):
        target = ast.literal_eval(read(spec['target']))
        size = np.array(ast.literal_eval(read(spec['size'])))
        threshold = ast.literal_eval(read(spec['threshold']))

        curr_positions = list(range(784))

        y0s = np.array(ast.literal_eval(read(spec['pathY'])))

        for i in range(100):
            print('\n==============\n')

            if len(curr_positions) <= 0: break

            pathX = spec['pathX'] + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(pathX)))

            output_x0 = model.apply(x0)
            y0 = np.argmax(output_x0, axis=1)[0]

            print('Data {}\n'.format(i))
            print('x0 = {}'.format(x0))
            print('output_x0 = {}'.format(output_x0))
            print('y0 = {}'.format(y0))
            print('y0[i] = {}\n'.format(y0s[i]))

            if y0 != y0s[i] or y0 == target:
                print('Skip at data {}'.format(i))
                continue
            else:
                output_x0_prob = softmax(output_x0.reshape(-1))
                print('output_x0_prob = {}'.format(output_x0_prob))

                prev_positions = curr_positions.copy()
                curr_positions = list()

                for position in prev_positions:
                    print('\n==============\n')

                    backdoor_indexes = self.__get_backdoor_indexes(size, position, model.shape)
                    print('backdoor_indexes = {}\n'.format(backdoor_indexes))

                    if backdoor_indexes is None:
                        print('Skip position!')
                        continue

                    lw, up = x0.copy(), x0.copy()

                    for index in backdoor_indexes:
                        lw[index], up[index] = model.lower[index], model.upper[index]

                    x0_poly = Poly()
                    x0_poly.lw, x0_poly.up = lw, up
                    # just let x0_poly.le and x0_poly.ge is None

                    lst_poly = [x0_poly]
                    self.__run(model, 0, lst_poly)

                    output_lw, output_up = lst_poly[-1].lw, lst_poly[-1].up

                    up_target = output_lw.copy()
                    up_target[target] = output_up[target]

                    up_target_prob = softmax(up_target)

                    print('up_target_prob = {}\n'.format(up_target_prob))

                    if up_target_prob[target] - output_x0_prob[target] > threshold:
                        print('Detect backdoor!')
                        curr_positions.append(position)
                    else:
                        print('No backdoor!')

        if len(curr_positions) > 0:
            print('Detect backdoor with target = {}!'.format(target))
        else:
            print('No backdoor with target = {}!'.format(target))


    def __get_backdoor_indexes(self, size, position, shape):
        if len(shape) == 2:
            num_rows = int(math.sqrt(shape[-1]))
            num_cols = num_rows
        else:
            num_rows = shape[-2]
            num_cols = shape[-1]

        row_idx = int(position / num_cols)
        col_idx = position - row_idx * num_cols

        if row_idx + size[0] > num_rows or col_idx + size[1] > num_cols:
            return None

        indexes = []

        for i in range(size[0]):
            for j in range(size[1]):
                indexes.append(position + j)
            position += num_cols

        return indexes


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_backdoor(model, assertion, display)
