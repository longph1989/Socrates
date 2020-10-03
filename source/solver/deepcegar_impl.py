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

        x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
        x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

        print('x0_poly.lw = {}'.format(x0_poly.lw))
        print('x0_poly.up = {}'.format(x0_poly.up))

        print('x0_poly.lt = \n{}'.format(x0_poly.lt))
        print('x0_poly.gt = \n{}'.format(x0_poly.gt))

        # shape = [*model.shape[1:]]
        # print('shape = {}'.format(shape))
        #
        # x0_poly.lw.reshape(shape)
        # x0_poly.up.reshape(shape)
        #
        # x0_poly.lt.reshape(*shape, -1)
        # x0_poly.gt.reshape(*shape, -1)

        res, x = self.__validate_x0(model, x0_poly, y0)
        if not res:
            y = np.argmax(model.apply(x), axis=1)[0]

            print('True adversarial sample found!')
            print('x = {}'.format(x))
            print('y = {}'.format(y))

            return

        xi_poly_prev = x0_poly
        lst_poly = [x0_poly]
        res = self.__verify(model, x0_poly, y0, xi_poly_prev, 0, lst_poly)

        if res:
            print('The network is robust around x0!')
        else:
            print('Unknown!')


    def __verify(self, model, x0_poly, y0, xi_poly_prev, idx, lst_poly):
        print('\n########################')
        print('idx = {}\n'.format(idx))

        if idx == len(model.layers):
            x = xi_poly_prev
            no_neurons = len(x.lw)

            for lbl in range(no_neurons):
                print('lbl = {}'.format(lbl))

                if lbl != y0 and x.lw[y0] <= x.up[lbl]:
                    coefs_curr = np.zeros(no_neurons + 1)
                    coefs_curr[y0] = 1
                    coefs_curr[lbl] = -1
                    
                    for k, e in reversed(list(enumerate(lst_poly))):
                        lt_prev = e.lt
                        gt_prev = e.gt

                        no_e_ns = len(e.lw)

                        if k > 0:
                            coefs = np.zeros(no_e_ns + 1)

                            for i in range(no_e_ns):
                                if coefs_curr[i] > 0:
                                    coefs = coefs + coefs_curr[i] * gt_prev[i]
                                elif coefs_curr[i] < 0:
                                    coefs = coefs + coefs_curr[i] * lt_prev[i]

                            coefs[-1] = coefs[-1] + coefs_curr[-1]

                            coefs_curr = coefs
                        else:
                            value = 0

                            for i in range(no_e_ns):
                                if coefs_curr[i] > 0:
                                    value = value + coefs_curr[i] * x0_poly.lw[i]
                                elif coefs_curr[i] < 0:
                                    value = value + coefs_curr[i] * x0_poly.up[i]

                                value = value + coefs_curr[-1]

                            print('value = {}\n'.format(value))

                            if value < 0: return False

            return True
        else:
            xi_poly_curr = model.forward(xi_poly_prev, x0_poly, idx, lst_poly)
            lst_poly.append(xi_poly_curr)

            print('xi_poly_curr.lw = {}'.format(xi_poly_curr.lw))
            print('xi_poly_curr.up = {}'.format(xi_poly_curr.up))

            print('xi_poly_curr.lt = \n{}'.format(xi_poly_curr.lt))
            print('xi_poly_curr.gt = \n{}'.format(xi_poly_curr.gt))

            return self.__verify(model, x0_poly, y0, xi_poly_curr, idx + 1, lst_poly)

            # if model.layers[idx].is_poly_exact():
            #     return self.__verify(model, x0_poly, y0, xi_poly_curr, idx + 1)
            #
            # res, x = self.__validate(model, x0_poly, y0, xi_poly_curr, idx + 1)
            #
            # if not res:
            #     # a counter example is found, should be fake
            #     print('Fake adversarial sample found!')
            #
            #     if self.count_ref >= self.max_ref:
            #         return False
            #     else:
            #         self.count_ref = self.count_ref + 1
            #
            #     len0 = len(x0_poly.lw)
            #     x = x[-len0:]
            #
            #     x_tmp = model.apply_to(x, idx)
            #
            #     g = grad(model.apply_from)(x_tmp, idx, y0=y0)
            #     ref_idx = np.argmax(g, axis=1)[0]
            #
            #     func = model.layers[idx].func
            #
            #     xi_poly_prev1, xi_poly_prev2 = self.__refine(xi_poly_prev, ref_idx, x_tmp, func)
            #
            #     if self.__verify(model, x0_poly, y0, xi_poly_prev1, idx):
            #         return self.__verify(model, x0_poly, y0, xi_poly_prev2, idx)
            #     else:
            #         return False
            # else:
            #     # ok, continue
            #     return self.__verify(model, x0_poly, y0, xi_poly_curr, idx + 1)


    def __refine(self, x_poly, idx, x_tmp, func):
        x1_poly = Poly()
        x2_poly = Poly()

        x1_poly.lw = x_poly.lw
        x1_poly.up = x_poly.up
        x1_poly.lt = x_poly.lt
        x1_poly.gt = x_poly.gt

        x2_poly.lw = x_poly.lw
        x2_poly.up = x_poly.up
        x2_poly.lt = x_poly.lt
        x2_poly.gt = x_poly.gt

        if func == relu:
            x1_poly.up[idx] = 0
            x2_poly.lw[idx] = 0
        elif func == sigmoid:
            x1_poly.up[idx] = x_tmp[idx]
            x2_poly.lw[idx] = x_tmp[idx]
        elif func == tanh:
            x1_poly.up[idx] = x_tmp[idx]
            x2_poly.lw[idx] = x_tmp[idx]

        return x1_poly, x2_poly


    def __validate_x0(self, model, x0_poly, y0):
        len0 = len(x0_poly.lw)

        x = np.zeros(len0)
        args = (model, y0)
        jac = grad(self.__obj_func_x0)

        bounds = Bounds(x0_poly.lw, x0_poly.up)

        res = minimize(self.__obj_func_x0, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func_x0(self, x, model, y0):
        output = model.apply(x)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)


    def __generate_constrains(self, coefs):
        def fun(x, coefs=coefs):
            res = 0
            lenx = len(x)
            for i in range(lenx):
                res = res + coefs[i] * x[i]
            res = res + coefs[lenx]
            return res
        return fun


    def __validate(self, model, x0_poly, y0, xi_poly, idx):
        len0 = len(x0_poly.lw)
        leni = len(xi_poly.lw)

        x = np.zeros(len0 + leni)
        args = (model, len0, leni, y0, idx)
        jac = grad(self.__obj_func)

        lw = np.concatenate([xi_poly.lw, x0_poly.lw])
        up = np.concatenate([xi_poly.up, x0_poly.up])
        bounds = Bounds(lw, up)

        lt = np.eye(leni) * (-1)
        gt = np.eye(leni)

        lt = np.concatenate([lt, xi_poly.lt], axis=1)
        gt = np.concatenate([gt, xi_poly.gt], axis=1)

        constraints = list()
        for coefs in lt:
            fun = self.__generate_constrains(coefs)
            constraints.append({'type': 'ineq', 'fun': fun})
        for coefs in gt:
            fun = self.__generate_constrains(coefs * (-1))
            constraints.append({'type': 'ineq', 'fun': fun})

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds, constraints=constraints)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func(self, x, model, len0, leni, y0, idx):
        x0 = x[-len0:]
        xi = x[:leni]

        tmp = model.apply_to(x0, idx)

        xi = xi.reshape(tmp.shape)
        output = model.apply_from(xi, idx)
        y0_score = output[0][y0]

        print('output = {}'.format(output))

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
