import numpy as np


class DeepCegarImpl():
    def __init__(self):
        pass


    def __solve_local_robustness(self, model, spec, display):
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]

        for idx in range(len(model.layers)):
            x0to = model.apply_to(x0, idx)
            res, x = self.__validate(model, spec, x0to, y0, idx)

            # need to transform to real x

            if not res:
                y = np.argmax(model.apply(x), axis=1)[0]
                if y0 != y:
                    print('True adversarial sample found!')
                    break
                else:
                    print('Fake adversarial sample found!')


    def __validate(self, model, spec, x0, y0, idx):
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

        x = x0.copy()
        args = (model, x0, y0, dfunc, eps, idx)
        bounds = Bounds(lower, upper)
        jac = grad(self.__obj_func) if model.layers != None else None

        res = minimize(self.__obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun == 0: # an adversarial sample is generated
            return False, res.x
        else:
            return True, np.empty(0)


    def __obj_func(self, x, model, x0, y0, dfunc, eps, idx):
        loss1 = dfunc(x, x0)
        loss1 = 0 if loss1 <= eps else loss1 - eps

        output = model.apply_from(x, idx)
        y0_score = output[0][y0]

        output = output - np.eye(output[0].size)[y0] * 1e9
        max_score = np.max(output)

        loss2 = 0 if y0_score < max_score else y0_score - max_score + 1e-9

        loss = loss1 + loss2

        return loss + np.sum(x - x)


    def solve(self, model, assertion, display=None):
        # only solve for local robustness
        return self.__solve_local_robustness(model, spec, display)
