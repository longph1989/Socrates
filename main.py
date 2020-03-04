import json
import ast
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import Bounds


def generate_adversarial_samples(spec):
    if 'robustness' in spec:
        generate_robustness(spec)
    else:
        generate_general(spec)


def generate_robustness(spec):
    size = spec['input']
    bnds = get_bounds(spec)
    model = get_model(spec)

    if spec['robustness'] == 'local':
        x0 = np.array(ast.literal_eval(" ".join(spec['origin'])))
    else:
        x0 = np.random.rand(size) * 2 - 1

    if 'distance' in spec:
        distance = spec['distance']
    else:
        distance = 'll_2'

    cons = list()
    if 'fairness' in spec:
        for ind in ast.literal_eval(spec['fairness']):
            type = 'eq'
            val = x0[ind]
            fun = get_fairness(ind, val)
            cons.append({'type': type, 'fun': fun})

    if 'target' in spec:
        target = spec['target']
        args = (size, model, x0, target, distance)

        optimize_robustness(args, bnds, cons)
    else:
        for i in range(spec['output']):
            target = i
            args = (size, model, x0, target, distance)

            optimize_robustness(args, bnds, cons)


def generate_general(spec):
    size = spec['input']
    bnds = get_bounds(spec)

    in_cons = list()
    for con in spec['in_cons']:
        type = con['type']
        coef = ast.literal_eval(con['coef'])
        fun = get_constraints(coef)
        in_cons.append({'type': type, 'fun': fun})

    out_cons = spec['out_cons']

    args = (size, model, out_cons)
    optimize_general(args, bnds, in_cons)


def get_model(spec):
    if 'model' in spec:
        model = torch.load(spec['model'])
    else:
        model = generate_model(spec)

    return model


def generate_model(spec):
    def fun(x_nn):
        x = x_nn.numpy()

        ws = list()
        bs = list()
        fs = list()

        steps = spec['steps']

        for step in steps:
            w = np.array(ast.literal_eval(step['weight']))
            b = np.array(ast.literal_eval(step['bias']))
            f = step['func']

            ws.append(w)
            bs.append(b)
            fs.append(f)

        for i in range(len(steps)):
            res = np.matmul(x, ws[i]) + bs[i]
            if fs[i] == 'relu':
                res = np.maximum(0, res)
            elif fs[i] == 'tanh':
                res = np.tanh(res)

        size = len(res[0])
        return torch.from_numpy(res).view(1, size)
    return fun


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


def get_fairness(ind, val):
    def fun(x, ind=ind, val=val):
        return x[ind] - val
    return fun


def get_constraints(coef):
    def fun(x, coef=coef):
        sum = 0
        size = len(coef) - 1
        for i in range(size):
            sum += coef[i] * x[i]
        sum +=coef[size]
        return sum
    return fun


def optimize_robustness(args, bnds, cons):
    size = args[0]     # size
    model = args[1]    # model

    x = args[2].copy() # x0

    res = minimize(func_robustness, x, args=args, bounds=bnds, constraints=cons)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        output = model(torch.from_numpy(res.x).view(1, size))
        print(output)


def func_robustness(x, size, model, x0, target, distance):
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


def optimize_general(args, bnds, cons):
    size = args[0]  # size
    model = args[1] # model

    x = np.zeros(size)

    res = minimize(func_general, x, args=args, bounds=bnds, constraints=cons)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        output = model(torch.from_numpy(res.x).view(1, size))
        print(output)


def func_general(x, size, model, cons):
    x_nn = torch.from_numpy(x).view(1, size)

    with torch.no_grad():
        output_x = model(x_nn)

    size = len(output_x[0])
    loss = 0

    for con in cons:
        type = con['type']
        coef = ast.literal_eval(con['coef'])

        sum = 0
        for i in range(size):
            sum += coef[i] * output_x[0][i]
        sum +=coef[size]

        if type == 'eq':
            loss_i = 0 if sum == 0 else abs(sum)
        elif type == 'ineq':
            loss_i = 0 if sum > 0 else abs(sum)

        loss = loss + loss_i

    return loss


def main():
    with open('spec.json', 'r') as f:
        spec = json.load(f)

    generate_adversarial_samples(spec)


if __name__ == '__main__':
    main()
