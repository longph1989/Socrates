import autograd.numpy as np
import ast

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import d0, d2, di
from utils import *


class Optimize():
    def __solve_syntactic_sugar(self, model, spec, display):
        if spec['robustness'] == 'local':
            self.__solve_local_robustness(model, spec, display)
        elif spec['robustness'] == 'global':
            self.__solve_global_robustness(model, spec)


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        res, x = self.__solve_robustness(model, spec, x0, y0)

        if not res and display:
            y = np.argmax(model.apply(x), axis=1)[0]
            display.show(model, x0, y0, x, y)


    def __solve_global_robustness(self, model, spec):
        n = 1000

        for i in range(n):
            x0 = self.__generate_x(model.shape, model.lower, model.upper)
            y0 = np.argmax(model.apply(x0), axis=1)[0]

            res, _ = self.__solve_robustness(model, spec, x0, y0)

            if not res:
                print('The model is unsatisfied with global robustness.')
                return

        print('The model is probably satisfied with global robustness after {} tests.'.format(n))


    def __solve_robustness(self, model, spec, x0, y0):
        lower = model.lower
        upper = model.upper

        eps = ast.literal_eval(read(spec['eps']))

        if spec['distance'] == 'd0':
            dfunc = d0
        elif spec['distance'] == 'd2':
            dfunc = d2
        elif spec['distance'] == 'di':
            dfunc = di
            lower = np.maximum(lower, x0 - eps)
            upper = np.minimum(upper, x0 + eps)

        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(x0.size):
                if not (index in sensitive):
                    lower[index] = x0[index]
                    upper[index] = x0[index]

        x = x0.copy()
        args = (model, x0, y0, dfunc, eps)
        bounds = Bounds(lower, upper)
        jac = grad(self.__obj_robustness) if model.layers != None else None

        res = minimize(self.__obj_robustness, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            print('The model is not robust around x0.')

            output_x = model.apply(res.x)
            lbl_x = np.argmax(output_x, axis=1)[0]

            print('x = {}'.format(res.x))
            print('output_x = {}'.format(output_x))
            print('lbl_x = {}'.format(lbl_x))

            return False, res.x
        else:
            print('The model is probably robust around x0.')
            return True, np.empty(0)


    def __obj_robustness(self, x, model, x0, y0, dfunc, eps):
        loss1 = dfunc(x, x0)
        loss1 = 0 if loss1 <= eps else loss1 - eps

        output = model.apply(x)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss2 = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        loss = loss1 + loss2

        return loss + np.sum(x - x)


    def __obj_func(self, x, model, assertion):
        vars_dict = dict()
        size = np.prod(model.shape)

        for i in range(len(assertion.vars)):
            var = assertion.vars[i]
            vars_dict[var.name] = x[i * size : (i + 1) * size]

        vars_dict.update(assertion.init_dict)

        return assertion.neg_num_value(vars_dict) + np.sum(x - x)


    def solve(self, model, assertion, display=None):
        if isinstance(assertion, dict):
            return self.__solve_syntactic_sugar(model, assertion, display)

        x = np.zeros(np.prod(model.shape) * len(assertion.vars))

        args = (model, assertion)
        bounds = Bounds(model.lower, model.upper)
        jac = grad(self.__obj_func) if model.layers != None else None

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0:
            print('The assertion is unsatisfied.'.format(res.x))

            output_x = model.apply(res.x)

            print('x = {}'.format(res.x))
            print('output_x = {}'.format(output_x))
        else:
            print('The assertion is probably satisfied.')



class SPRT():
    def __init__(self, threshold, alpha, beta, delta):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.delta = delta


    def __generate_x(self, shape, lower, upper):
        size = np.prod(shape)
        x = np.random.rand(size)

        x = (upper - lower) * x + lower

        return x


    def __solve_syntactic_sugar(self, model, spec):
        if spec['robustness'] == 'local':
            self.__solve_local_robustness(model, spec)
        elif spec['robustness'] == 'global':
            self.__solve_global_robustness(model, spec)


    def __solve_local_robustness(self, model, spec):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        self.__solve_robustness(model, spec, x0, y0)


    def __solve_global_robustness(self, model, spec):
        n = 1000

        for i in range(n):
            x0 = self.__generate_x(model.shape, model.lower, model.upper)
            y0 = np.argmax(model.apply(x0), axis=1)[0]

            if not self.__solve_robustness(model, spec, x0, y0):
                print('The model is unsatisfied with global robustness.')
                return

        print('The model is probably satisfied with global robustness after {} tests.'.format(n))


    def __solve_robustness(self, model, spec, x0, y0):
        lower = model.lower
        upper = model.upper

        eps = ast.literal_eval(read(spec['eps']))

        if spec['distance'] == 'd0':
            dfunc = d0
        elif spec['distance'] == 'd2':
            dfunc = d2
        elif spec['distance'] == 'di':
            dfunc = di
            lower = np.maximum(lower, x0 - eps)
            upper = np.minimum(upper, x0 + eps)

        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(x0.size):
                if not (index in sensitive):
                    lower[index] = x0[index]
                    upper[index] = x0[index]

        p0 = self.threshold + self.delta
        p1 = self.threshold - self.delta

        h0 = self.beta / (1 - self.alpha)
        h1 = (1 - self.beta) / self.alpha

        pr = 1
        no = 0

        while True:
            x = self.__generate_x(model.shape, lower, upper)

            if dfunc(x, x0) <= eps:
                no = no + 1

                y = np.argmax(model.apply(x), axis=1)[0]

                if y == y0:
                    pr = pr * p1 / p0
                else:
                    pr = pr * (1 - p1) / (1 - p0)

                if pr <= h0:
                    print('Accept H0. The model is robust around x0 with p >= {} after {} tests.'.format(p0, no))
                    return True
                elif pr >= h1:
                    print('Accept H1. The model is robust around x0 with p <= {} after {} tests.'.format(p1, no))
                    return False


    def solve(self, model, assertion, display=None):
        if isinstance(assertion, dict):
            return self.__solve_syntactic_sugar(model, assertion)

        p0 = self.threshold + self.delta
        p1 = self.threshold - self.delta

        h0 = self.beta / (1 - self.alpha)
        h1 = (1 - self.beta) / self.alpha

        pr = 1
        no = 0

        while True:
            vars_dict = dict()

            for var in assertion.vars:
                x = self.__generate_x(model.shape, model.lower, model.upper)
                vars_dict[var.name] = x

            vars_dict.update(assertion.init_dict)

            if assertion.get_pre_bool_value(vars_dict):
                no = no + 1

                if assertion.get_post_bool_value(vars_dict):
                    pr = pr * p1 / p0
                else:
                    pr = pr * (1 - p1) / (1 - p0)

                if pr <= h0:
                    print('Accept H0. The assertion is satisfied with p >= {} after {} tests.'.format(p0, no))
                    break
                elif pr >= h1:
                    print('Accept H1. The assertion is satisfied with p <= {} after {} tests.'.format(p1, no))
                    break
