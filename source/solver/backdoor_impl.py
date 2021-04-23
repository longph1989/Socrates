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

        dataset = spec['dataset']

        valid_x0s = []
        y0s = np.array(ast.literal_eval(read(spec['pathY'])))

        for i in range(100):
            # print('\n==============\n')

            pathX = spec['pathX'] + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(pathX)))

            output_x0 = model.apply(x0).reshape(-1)
            y0 = np.argmax(output_x0)

            # print('Data {}\n'.format(i))
            # print('x0 = {}'.format(x0))
            # print('output_x0 = {}'.format(output_x0))
            # print('y0 = {}'.format(y0))
            # print('y0[i] = {}\n'.format(y0s[i]))

            if y0 == y0s[i] and y0 != target and softmax(output_x0)[target] < 0.01:
                # print('Check at data {}'.format(i))
                valid_x0s.append((x0, output_x0))
            else:
                # print('Skip at data {}'.format(i))
                pass

        if len(valid_x0s) == 0: return

        if dataset == 'mnist': positions = 784
        elif dataset == 'cifar': positions = 1024

        for position in range(positions):
            # print('\n==============\n')

            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)
            # print('backdoor_indexes = {}\n'.format(backdoor_indexes))

            if backdoor_indexes is None:
                # print('Skip position!')
                continue

            cnt = 0

            for x0, output_x0 in valid_x0s:
                lw, up = x0.copy(), x0.copy()

                for index in backdoor_indexes:
                    lw[index], up[index] = model.lower[index], model.upper[index]

                x0_poly = Poly()
                x0_poly.lw, x0_poly.up = lw, up
                # just let x0_poly.le and x0_poly.ge is None
                x0_poly.shape = model.shape

                lst_poly = [x0_poly]
                self.__run(model, 0, lst_poly)

                output_lw, output_up = lst_poly[-1].lw.copy(), lst_poly[-1].up.copy()
                output_lw[target] = output_up[target]

                if softmax(output_lw)[target] / softmax(output_x0)[target] < 95:
                    # print('No backdoor at {}!'.format(cnt))
                    break
                else:
                    cnt += 1

            if cnt == len(valid_x0s):
                # print('Detect backdoor with target = {} and position = {}!'.format(target, position))
                # self.__validate(model, valid_x0s, target, backdoor_indexes)
                return target

        # print('No backdoor with target = {}!'.format(target))
        return None


    # def __validate(self, model, valid_x0s, target, backdoor_indexes):
    #     def obj_func(x, model, valid_x0s, target, backdoor_indexes):
    #         res = 0

    #         for x0, output_x0 in valid_x0s:
    #             x0[backdoor_indexes] = x

    #             output = model.apply(x0).reshape(-1)
    #             target_score = output[target]

    #             output_no_target = output - np.eye(len(output))[target] * 1e9
    #             max_score = np.max(output_no_target)

    #             if target_score > max_score: res += 0
    #             else: res += max_score - target_score

    #         return res

    #     x = np.zeros(len(backdoor_indexes))
    #     lw = model.lower[backdoor_indexes]
    #     up = model.upper[backdoor_indexes]

    #     args = (model, valid_x0s, target, backdoor_indexes)
    #     # jac = grad(obj_func)
    #     jac = None
    #     bounds = Bounds(lw, up)

    #     res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

    #     if res.fun <= 0: # an adversarial sample is generated
    #         print('res.x = {}'.format(res.x))
    #         return res.x

    #     return None


    def __get_backdoor_indexes(self, size, position, dataset):
        if dataset == 'mnist':
            num_chans, num_rows, num_cols = 1, 28, 28
        elif dataset == 'cifar':
            num_chans, num_rows, num_cols = 3, 32, 32

        row_idx = int(position / num_cols)
        col_idx = position - row_idx * num_cols

        if row_idx + size[0] > num_rows or col_idx + size[1] > num_cols:
            return None

        indexes = []

        for i in range(num_chans):
            tmp = position + i * num_rows * num_cols
            for j in range(size[0]):
                for k in range(size[1]):
                    indexes.append(tmp + k)
                tmp += num_cols

        return indexes


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        return self.__solve_backdoor(model, assertion, display)
