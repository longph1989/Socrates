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
    def __init__(self):
        pass


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))

        x0_poly = Poly()

        x0_poly.lw = np.maximum(model.lower, x0 - eps)
        x0_poly.up = np.minimum(model.upper, x0 + eps)

        x0_poly.lt = np.eye(len(x0) + 1)[0:-1]
        x0_poly.gt = np.eye(len(x0) + 1)[0:-1]

        res, x = self.__validate_x0(model, x0_poly, y0)
        if not res:
            print('True adversarial sample found!')
            return

        xi_poly_prev = x0_poly
        res = self.__verify(model, x0_poly, y0, xi_poly_prev, 0)

        if res:
            print('The network is robust around x0!')
        else:
            print('Unknown!')


    def __verify(self, model, x0_poly, y0, xi_poly_prev, idx):
        print(idx)
        print(xi_poly_prev.lw.shape)
        print(xi_poly_prev.up.shape)
        print(xi_poly_prev.lt.shape)
        print(xi_poly_prev.gt.shape)

        if idx == len(model.layers):
            no_features = len(x0_poly.lw)
            x = xi_poly_prev

            for i in range(len(x.up)):
                if i != y0 and x.lw[y0] <= x.up[i]:
                    coefs = x.gt[y0] - x.lt[i]
                    lower = 0

                    for i in range(no_features):
                        if coefs[i] > 0:
                            lower = lower + coefs[i] * x0_poly.lw[i]
                        else:
                            lower = lower + coefs[i] * x0_poly.up[i]

                    lower = lower + coefs[-1]

                    if lower < 0: return False

            return True
        else:
            xi_poly_curr = model.forward(xi_poly_prev, x0_poly, idx)
            res, x = self.__validate(model, x0_poly, y0, xi_poly_curr, idx)

            if not res:
                # a counter example is found, should be fake
                print('Fake adversarial sample found!')

                len0 = len(x0_poly_curr.lw)
                x = x[-len0:]

                x_tmp = model.apply_to(x, idx)

                # y = np.argmax(model.apply_from(x_tmp), axis=1)[0]
                #
                # if y0 != y:
                #     print('True adversarial sample found!')
                #     return False
                # else:

                g = grad(model.apply_from)(x_tmp, idx)
                ref_idx = np.argmax(g, axis=1)[0]

                xi_poly_prev1, xi_poly_prev2 = self.__refine(xi_poly_prev, x_tmp, ref_idx)

                if self.__verify(model, x0_poly, y0, xi_poly_prev1, idx):
                    return self.__verify(model, x0_poly, y0, xi_poly_prev2, idx)
                else:
                    return False
            else:
                # ok, continue
                return self.__verify(model, x0_poly, y0, xi_poly_curr, idx + 1)


    def __refine(x_poly, x, idx):
        # try for relu first

        x1_poly = Poly()
        x1_poly.lw = x_poly.lw
        x1_poly.up = x_poly.up
        x1_poly.lt = x_poly.lt
        x1_poly.gt = x_poly.gt

        x2_poly = Poly()
        x2_poly.lw = x_poly.lw
        x2_poly.up = x_poly.up
        x2_poly.lt = x_poly.lt
        x2_poly.gt = x_poly.gt

        x1_poly.up[idx] = 0
        x2_poly.lw[idx] = 0

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
        return True, np.empty(0)

        len0 = len(x0_poly.lw)
        leni = len(xi_poly.lw)

        x = np.zeros(len0 + leni)
        args = (model, leni, y0, idx)
        jac = grad(self.__obj_func)

        lw = np.concatenate([xi_poly.lw, x0_poly.lw])
        up = np.concatenate([xi_poly.up, x0_poly.up])
        bounds = Bounds(lw, up)

        lt = np.eye(leni)
        gt = np.eye(leni)

        lt = np.concatenate([lt, xi_poly.lt], axis=1)
        gt = np.concatenate([gt, xi_poly.gt], axis=1)

        constraints = list()
        for coefs in lt:
            fun = self.__generate_constrains(coefs * (-1))
            constraints.append({'type': 'ineq', 'fun': fun})
        for coefs in gt:
            fun = self.__generate_constrains(coefs)
            constraints.append({'type': 'ineq', 'fun': fun})

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds, constraints=constraints)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func(self, x, model, leni, y0, idx):
        xi = x[:leni]
        output = model.apply_from(xi, idx)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, assertion, display)
