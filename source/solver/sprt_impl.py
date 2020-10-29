import autograd.numpy as np
import ast

from autograd import grad
from assertion.lib_functions import d0, d2, di
from utils import *


class SPRTImpl():
    def __init__(self, threshold, alpha, beta, delta):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.delta = delta


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
        
        size = np.prod(model.shape)

        for i in range(n):
            x0 = generate_x(size, model.lower, model.upper)
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
        
        size = np.prod(model.shape)

        while True:
            x = generate_x(size, lower, upper)

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
