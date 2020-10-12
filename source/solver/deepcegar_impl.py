import autograd.numpy as np
import ast

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *


class Poly():
    def __init__(self):
        self.lw = None
        self.up = None

        self.lt = None
        self.gt = None

    def back_substitute_bounds(self, lst_poly):
        lt_curr = self.lt
        gt_curr = self.gt

        no_curr_ns = len(lt_curr)

        lw = np.zeros(no_curr_ns)
        up = np.zeros(no_curr_ns)

        for k, e in reversed(list(enumerate(lst_poly))):
            lt_prev = e.lt
            gt_prev = e.gt

            no_e_ns = len(lt_prev)
            no_coefs = len(lt_prev[0])

            if k > 0:
                lt = np.zeros([no_curr_ns, no_coefs])
                gt = np.zeros([no_curr_ns, no_coefs])

                for i in range(no_curr_ns):
                    for j in range(no_e_ns):
                        if lt_curr[i,j] > 0:
                            lt[i] = lt[i] + lt_curr[i,j] * lt_prev[j]
                        elif lt_curr[i,j] < 0:
                            lt[i] = lt[i] + lt_curr[i,j] * gt_prev[j]

                        if gt_curr[i,j] > 0:
                            gt[i] = gt[i] + gt_curr[i,j] * gt_prev[j]
                        elif gt_curr[i,j] < 0:
                            gt[i] = gt[i] + gt_curr[i,j] * lt_prev[j]

                    lt[i,-1] = lt[i,-1] + lt_curr[i,-1]
                    gt[i,-1] = gt[i,-1] + gt_curr[i,-1]

                lt_curr = lt
                gt_curr = gt
            else:
                for i in range(no_curr_ns):
                    for j in range(no_e_ns):
                        if lt_curr[i,j] > 0:
                            up[i] = up[i] + lt_curr[i,j] * e.up[j]
                        elif lt_curr[i,j] < 0:
                            up[i] = up[i] + lt_curr[i,j] * e.lw[j]

                        if gt_curr[i,j] > 0:
                            lw[i] = lw[i] + gt_curr[i,j] * e.lw[j]
                        elif gt_curr[i,j] < 0:
                            lw[i] = lw[i] + gt_curr[i,j] * e.up[j]

                    up[i] = up[i] + lt_curr[i,-1]
                    lw[i] = lw[i] + gt_curr[i,-1]

        self.lw = lw
        self.up = up


class DeepCegarImpl():
    def __init__(self, max_ref):
        self.count_ref = 0
        self.max_ref = max_ref


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))

        print('x0 = {}'.format(x0))
        print('y0 = {}'.format(y0))

        x0_poly = Poly()

        x0_poly.lw = np.maximum(model.lower, x0 - eps)
        x0_poly.up = np.minimum(model.upper, x0 + eps)

        # print('x0_poly.lw = {}'.format(x0_poly.lw))
        # print('x0_poly.up = {}'.format(x0_poly.up))

        x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
        x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

        # print('x0_poly.lt = \n{}'.format(x0_poly.lt))
        # print('x0_poly.gt = \n{}'.format(x0_poly.gt))

        lst_poly = [x0_poly]
        res = self.__verify(model, y0, x0_poly, 0, lst_poly)

        if res:
            print('The network is robust around x0!')
        else:
            print('Unknown!')


    def __verify(self, model, y0, xi_poly_prev, idx, lst_poly):
        # print('\n########################')
        # print('idx = {}\n'.format(idx))

        if idx == len(model.layers):
            x = xi_poly_prev
            no_neurons = len(x.lw)

            print('lw = {}'.format(x.lw))
            print('up = {}'.format(x.up))

            for lbl in range(no_neurons):
                # print('lbl = {}'.format(lbl))

                if lbl != y0 and x.lw[y0] <= x.up[lbl]:
                    res_poly = Poly()

                    coefs_curr = np.zeros(no_neurons + 1)
                    coefs_curr[y0] = 1
                    coefs_curr[lbl] = -1

                    res_poly.lt = np.zeros([1, no_neurons + 1])
                    res_poly.gt = np.zeros([1, no_neurons + 1])

                    res_poly.gt[0,y0] = 1
                    res_poly.gt[0,lbl] = -1

                    res_poly.back_substitute_bounds(lst_poly)

                    # print('res_lw = {}\n'.format(res_poly.lw))

                    if res_poly.lw < 0: return False

            return True
        else:
            xi_poly_curr = model.forward(xi_poly_prev, idx, lst_poly)

            # print('xi_poly_curr.lw = {}'.format(xi_poly_curr.lw))
            # print('xi_poly_curr.up = {}'.format(xi_poly_curr.up))
            #
            # print('xi_poly_curr.lt = \n{}'.format(xi_poly_curr.lt))
            # print('xi_poly_curr.gt = \n{}'.format(xi_poly_curr.gt))

            lst_poly.append(xi_poly_curr)
            return self.__verify(model, y0, xi_poly_curr, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
