import json
import ast
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import Bounds


def get_bounds(spec):
    size = spec['size']

    if 'bounds' in spec:
        bnds = spec['bounds']
        if type(bnds) == list:
            lw = np.zeros(size)
            up = np.zeros(size)
            for i in range(size):
                bnd = ast.literal_eval(bnds[i])
                lw[i] = bnd[0]
                up[i] = bnd[1]
        else:
            bnds = ast.literal_eval(bnds)
            lw = np.full(size, bnds[0])
            up = np.full(size, bnds[1])
    else:
        lw = np.full(size, -1.0)
        up = np.full(size, 1.0)

    return Bounds(lw, up)


def main():
    with open('spec.json', 'r') as f:
        spec = json.load(f)

    shape = ast.literal_eval(spec['input'])
    model = torch.load(spec['model'])

    bounds = get_bounds(spec)

    if 'robustness' in spec:
        if spec['robustness'] == 'local':
            x0 = np.array(ast.literal_eval(" ".join(spec['origin'])))

            if 'target' in spec:
                x = x0.copy()

                target = int(spec['target'])
                args = (x0, shape, model, target)

                transform(x, args, bounds)
            else:
                for i in range(spec['output']):
                    x = x0.copy()

                    target = i
                    args = (x0, shape, model, target)

                    transform(x, args, bounds)
        else:
            x0 = np.random.rand(shape) * 2 - 1

            for i in range(spec['output']):
                x = x0.copy()

                target = i
                args = (x0, shape, model, target)

                transform(x, args, bounds)
    else:
        print('fuck you')


def transform(x, args, bounds):
    res = minimize(func, x, args=args, bounds=bounds)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        shape = args[1]
        model = args[2]
        output = model(torch.from_numpy(res.x).view(shape))
        print(output)


def func(x, x0, shape, model, target):
    # loss1 = np.sqrt(np.sum((x - x0) ** 2))
    # loss1 = np.sum(x != x0)
    loss1 = np.max(np.absolute(x - x0))

    x_nn = torch.from_numpy(x).view(shape)

    with torch.no_grad():
        output_x = model(x_nn)

    target_score = output_x[0][target]
    max_score = torch.max(output_x)

    loss2 = 0 if target_score >= max_score else max_score - target_score

    loss = loss1 + loss2

    return loss

if __name__ == '__main__':
    main()
