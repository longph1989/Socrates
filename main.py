import json
import ast
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import Bounds


def get_bounds(spec):
    size = spec['input']

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


def generate_adversarial_samples(spec):
    size = spec['input']
    bounds = get_bounds(spec)

    model = torch.load(spec['model'])

    if 'robustness' in spec:
        if spec['robustness'] == 'local':
            x0 = np.array(ast.literal_eval(" ".join(spec['origin'])))
        else:
            x0 = np.random.rand(1, size) * 2 - 1

        if 'distance' in spec:
            distance = spec['distance']
        else:
            distance = 'll_2'

        if 'target' in spec:
            target = spec['target']
            args = (x0, size, model, target, distance)

            transform(args, bounds)
        else:
            for i in range(spec['output']):
                target = i
                args = (x0, size, model, target, distance)

                transform(args, bounds)
    else:
        print('fuck you')


def transform(args, bounds):
    x = args[0].copy() # x0

    res = minimize(func, x, args=args, bounds=bounds)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        size = args[1]  # size
        model = args[2] # model

        output = model(torch.from_numpy(res.x).view(1, size))
        print(output)


def func(x, x0, size, model, target, distance):
    if distance == 'll_0':
        loss1 = np.sum(x != x0)
    elif distance == 'll_2':
        loss1 = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        loss1 = np.max(np.absolute(x - x0))
    else:
        loss1 = 0

    x_nn = torch.from_numpy(x).view(1, size)

    with torch.no_grad():
        output_x = model(x_nn)

    target_score = output_x[0][target]
    max_score = torch.max(output_x)

    loss2 = 0 if target_score >= max_score else max_score - target_score

    loss = loss1 + loss2

    return loss


def main():
    with open('spec.json', 'r') as f:
        spec = json.load(f)

    generate_adversarial_samples(spec)


if __name__ == '__main__':
    main()
