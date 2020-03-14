import argparse
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
    size = spec['in_size']
    shape = ast.literal_eval(spec['in_shape'])

    lb, ub = get_bounds(spec)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    if spec['robustness'] == 'local':
        print('Local robustness\n')
        x0 = np.array(ast.literal_eval(" ".join(spec['origin'])))
    else:
        print('Global robustness\n')
        x0 = np.random.rand(size) * 2 - 1
        for i in range(size):
            x0[i] = max(x0[i], lb[i])
            x0[i] = min(x0[i], ub[i])

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
        args = (shape, model, x0, target, distance)

        print('Target = {}\n'.format(target))

        optimize_robustness(args, bnds, cons)
    else:
        for i in range(spec['out_size']):
            target = i
            args = (shape, model, x0, target, distance)

            print('Target = {}\n'.format(target))

            optimize_robustness(args, bnds, cons)


def generate_general(spec):
    size = spec['in_size']
    shape = ast.literal_eval(spec['in_shape'])

    lb, ub = get_bounds(spec)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    x0 = np.zeros(size)

    in_cons = list()
    for con in spec['in_cons']:
        type = con['type']
        coef = ast.literal_eval(con['coef'])
        fun = get_constraints(coef)
        in_cons.append({'type': type, 'fun': fun})

    out_cons = spec['out_cons']

    args = (shape, model, x0, out_cons)
    optimize_general(args, bnds, in_cons)


def get_model(spec):
    if 'model' in spec:
        model = torch.load(spec['model'])
    else:
        ws = list()
        bs = list()
        fs = list()

        layers = spec['layers']

        for layer in layers:
            wt = open(layer['weight'], 'r').readline()
            bt = open(layer['bias'], 'r').readline()

            w = np.transpose(np.array(ast.literal_eval(wt)))
            b = np.expand_dims(np.array(ast.literal_eval(bt)), axis=0)
            f = layer['func']

            ws.append(w)
            bs.append(b)
            fs.append(f)

        model = generate_model(ws, bs, fs)

    return model


def generate_model(ws, bs, fs):
    def fun(x_nn, ws=ws, bs=bs, fs=fs):
        x = x_nn.numpy()
        res = x

        for i in range(len(ws)):
            res = np.matmul(res, ws[i]) + bs[i]
            if fs[i] == 'relu':
                res = np.maximum(0, res)
            elif fs[i] == 'tanh':
                res = np.tanh(res)

        size = len(res[0])
        return torch.from_numpy(res).view(1, size)
    return fun


def get_bounds(spec):
    size = spec['in_size']

    if 'bounds' in spec:
        bnds = spec['bounds']
        if type(bnds) == list:
            lb = np.zeros(size)
            ub = np.zeros(size)
            for i in range(size):
                bnd = ast.literal_eval(bnds[i])
                lb[i] = bnd[0]
                ub[i] = bnd[1]
        else:
            bnds = ast.literal_eval(bnds)
            lb = np.full(size, bnds[0])
            ub = np.full(size, bnds[1])
    else:
        lb = np.full(size, -1.0)
        ub = np.full(size, 1.0)

    return lb, ub


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
        sum += coef[size]
        return sum
    return fun


def optimize_robustness(args, bnds, cons):
    shape = args[0]    # shape
    model = args[1]    # model

    x = args[2].copy() # x0

    res = minimize(func_robustness, x, args=args, bounds=bnds, constraints=cons)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        output = model(torch.from_numpy(res.x).view(shape))
        print(output)


def func_robustness(x, shape, model, x0, target, distance):
    if distance == 'll_0':
        loss1 = np.sum(x != x0)
    elif distance == 'll_2':
        loss1 = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        loss1 = np.max(np.absolute(x - x0))
    else:
        loss1 = 0

    x_nn = torch.from_numpy(x).view(shape)

    with torch.no_grad():
        output_x = model(x_nn)

    target_score = output_x[0][target]
    max_score = torch.max(output_x)

    loss2 = 0 if target_score >= max_score else max_score - target_score

    loss = loss1 + loss2

    return loss


def optimize_general(args, bnds, cons):
    shape = args[0] # shape
    model = args[1] # model

    x = args[2]     # x0, no need to copy

    res = minimize(func_general, x, args=args, bounds=bnds, constraints=cons)

    print("Global minimum x0 = {}, f(x0) = {}".format(res.x.shape, res.fun))

    with torch.no_grad():
        output = model(torch.from_numpy(res.x).view(shape))
        print(output)


def func_general(x, shape, model, x0, cons):
    x_nn = torch.from_numpy(x).view(shape)

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

        loss += loss_i

    return loss


def main():
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification input file')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    generate_adversarial_samples(spec)


if __name__ == '__main__':
    main()
