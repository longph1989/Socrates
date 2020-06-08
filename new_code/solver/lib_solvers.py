import autograd.numpy as np

class Optimize():
    def __init__(self):
        return

    def solve(self, model, assertion):
        return


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
        return x

    def solve(self, model, assertion):
        if isinstance(assertion, dict):
            return self.__solve_syntactic_sugar(model, assertion)

        p0 = self.threshold + self.delta
        p1 = self.threshold - self.delta

        h0 = self.beta / (1 - self.alpha)
        h1 = (1 - self.beta) / self.alpha

        pr = 1

        while True:
            vars_dict = {}

            for var in assertion.vars:
                x = self.__generate_x(model.shape, model.lower, model.upper)
                vars_dict[var.name] = x

            if assertion.get_pre_value(vars_dict):
                if assertion.get_post_value(vars_dict):
                    pr = pr * p1 / p0
                else:
                    pr = pr * (1 - p1) / (1 - p0)

            if pr <= h0:
                print('Accept H0. The assertion is satisfied with p >= {} after {} tests.'.format(p0, no_x))
                break
            elif pr >= h1:
                print('Accept H1. The assertion is satisfied with p <= {} after {} tests.'.format(p1, no_x))
                break
