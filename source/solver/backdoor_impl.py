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
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        output_x0 = model.apply(x0)
        y0 = np.argmax(output_x0, axis=1)[0]

        # x0[0], x0[1], x0[28], x0[29] = 1, 1, 1, 1
        # x0[754], x0[755], x0[782], x0[783] = 1, 1, 1, 1

        # fig, ax = plt.subplots(1, 1)

        # ax.imshow((x0 * 255).astype('uint8').reshape(28,28), cmap='gray')
        # plt.show()

        # raise Exception()
        
        output_x0_prob = softmax(output_x0.reshape(-1))

        print('output_x0_prob = {}\n'.format(output_x0_prob))

        target = ast.literal_eval(read(spec['target']))
        threshold = ast.literal_eval(read(spec['threshold']))

        size = np.array(ast.literal_eval(read(spec['size'])))
        position = ast.literal_eval(read(spec['position']))

        backdoor_indexes = self.__get_backdoor_indexes(size, position, model.shape)
        print('backdoor_indexes = {}\n'.format(backdoor_indexes))

        lw, up = x0.copy(), x0.copy()

        for index in backdoor_indexes:
            if index < len(model.lower):
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
            return True
        else:
            print('No backdoor!')
            return False


    def __get_backdoor_indexes(self, size, position, shape):
        if len(shape) == 2:
            cols = int(math.sqrt(shape[-1]))
        else:
            cols = shape[-1]

        indexes = []

        for i in range(size[0]):
            for j in range(size[1]):
                indexes.append(position + j)
            position += cols

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
