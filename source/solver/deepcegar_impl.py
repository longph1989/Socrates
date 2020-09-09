import numpy as np

from assertion.lib_functions import di


class Poly():
    def __init__(x0, eps, lower, upper):
        self.lw = np.maximum(lower, x0 - eps)
        self.up = np.minimum(upper, x0 + eps)

        lt = np.eye(len(x0))
        self.lt = np.concatenate([lt, lw.reshape(-1, len(lw)).transpose() * (-1)], axis=1)

        gt = np.eye(len(x0))
        self.gt = np.concatenate([gt, up.reshape(-1, len(lw)).transpose() * (-1)], axis=1)


class DeepCegarImpl():
    def __init__(self):
        pass


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        eps = ast.literal_eval(read(spec['eps']))
        x0_poly = Poly(x0, eps, model.lower, model.upper)

        for idx in range(len(model.layers)):
            xt_poly = model.apply_to(x0_poly, idx)
            res, x = self.__validate(model, spec, xt_poly, y0, idx)

            if not res:
                y = np.argmax(model.apply(x), axis=1)[0]
                if y0 != y:
                    print('True adversarial sample found!')
                    break
                else:
                    print('Fake adversarial sample found!')


    def __generate_constrains(coefs):
        def fun(x, coefs=coefs):
            res = 0
            len = len(x)
            for i in range(len):
                res = res + coefs[i] * x[i]
            res = res + coefs[len]
            return res
        return fun


    def __validate(self, model, spec, xt_poly, y0, idx):
        x = np.zeros(len(xt_poly.lw))
        args = (model, y0, idx)
        bounds = Bounds(xt_poly.lw, xt_poly.up)
        jac = grad(self.__obj_func)

        constraints = list()
        for coefs in xt_poly.lt:
            fun = self.__generate_constrains(coefs * (-1))
            constraints.append({'type': 'ineq', 'fun': fun})
        for coefs in xt_poly.gt:
            fun = self.__generate_constrains(coefs)
            constraints.append({'type': 'ineq', 'fun': fun})

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds, constraints=constraints)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func(self, x, model, y0, idx):
        output = model.apply_from(x, idx) # seem problem
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        return loss + np.sum(x - x)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, spec, display)
