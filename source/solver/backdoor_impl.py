import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os
import time
import random

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *
from solver.deepcegar_impl import Poly


class BackDoorImpl():
    def __solve_backdoor(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        output_x0 = model.apply(x0)
        y0 = np.argmax(output_x0, axis=1)[0]

        output_x0_prob = softmax(output_x0.reshape(-1))

        print('output_x0_prob = {}'.format(output_x0_prob))

        lw, up = model.lower, model.upper

        for index in range(784):
            if index != 783:
                lw[index], up[index] = x0[index], x0[index]

        x0_poly = Poly()
        x0_poly.lw, x0_poly.up = lw, up
        # just let x0_poly.le and x0_poly.ge is None

        lst_poly = [x0_poly]
        self.__run(model, 0, lst_poly)

        output_lw, output_up = lst_poly[-1].lw, lst_poly[-1].up

        target = ast.literal_eval(read(spec['target']))

        up_target = output_lw.copy()
        up_target[target] = output_up[target]

        up_target_prob = softmax(up_target)

        print('up_target_prob = {}'.format(up_target_prob))          



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
