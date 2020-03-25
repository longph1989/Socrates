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
        x0t = open(spec['x0'], 'r').readline()
        x0 = np.array(ast.literal_eval(x0t))
    else:
        print('Global robustness\n')
        x0 = np.random.rand(size) * 2 - 1
        for i in range(size):
            x0[i] = max(x0[i], lb[i])
            x0[i] = min(x0[i], ub[i])

    h0 = None
    if 'h0' in spec:
        h0t = open(spec['h0'], 'r').readline()
        h0 = np.array(ast.literal_eval(h0t))

    if 'distance' in spec:
        distance = spec['distance']
    else:
        distance = 'll_i'

    cons = list()
    if 'fairness' in spec:
        for ind in ast.literal_eval(spec['fairness']):
            type = 'eq'
            val = x0[ind]
            fun = get_fairness(ind, val)
            cons.append({'type': type, 'fun': fun})

    print('x0 = {}'.format(x0))
    output_x0 = apply_model(model, x0, shape, h0)
    print('Original output = {}'.format(output_x0))

    if 'target' in spec:
        target = spec['target']
        args = (shape, model, x0, target, distance, h0)

        print('Target = {}\n'.format(target))

        optimize_robustness(args, bnds, cons)
    else:
        for i in range(spec['out_size']):
            target = i
            args = (shape, model, x0, target, distance, h0)

            print('Target = {}\n'.format(target))

            optimize_robustness(args, bnds, cons)


def generate_general(spec):
    size = spec['in_size']
    shape = ast.literal_eval(spec['in_shape'])

    lb, ub = get_bounds(spec)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    x0 = np.zeros(size)

    h0 = None
    if 'h0' in spec:
        h0t = open(spec['h0'], 'r').readline()
        h0 = np.array(ast.literal_eval(h0t))

    print('x0 = {}'.format(x0))
    output_x0 = apply_model(model, x0, shape, h0)
    print('Original output = {}'.format(output_x0))

    in_cons = list()

    if 'in_cons' in spec:
        for con in spec['in_cons']:
            type = con['type']
            coef = ast.literal_eval(con['coef'])
            fun = get_constraints(coef)
            in_cons.append({'type': type, 'fun': fun})

    out_cons = spec['out_cons']

    args = (shape, model, x0, out_cons, h0)
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
    len = ast.literal_eval(spec['in_shape'])[0]

    size = size * len

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


def apply_model(model, x, shape, h0):
    with torch.no_grad():
        if shape[0] == 1:
            x_nn = torch.from_numpy(x).view(shape)
            output_x = model(x_nn)
        else:
            shape_h = [1,h0.size]
            h = torch.from_numpy(h0).view(shape_h)
            shape_x = [1,*shape[1:]]
            size_x = shape[1]
            for i in range(shape[0]):
                x_nn = torch.from_numpy(x[size_x * i : size_x * (i + 1)]).view(shape_x)
                output_x, h = model(x_nn, h)
    return output_x


def optimize_robustness(args, bnds, cons):
    shape = args[0]    # shape
    model = args[1]    # model

    target = args[3]   # target

    cmin = 1
    cmax = 100

    best_x = []
    best_dist = 1e9

    while True:
        if cmin >= cmax: break

        c = int((cmin + cmax) / 2)
        print('c = {}'.format(c))

        x = args[2]    # x0
        h0 = args[-1]

        args_c = (*args, c)

        res = minimize(func_robustness, x, args=args_c, bounds=bnds, constraints=cons)

        print('Global minimum x0 = {}, f(x0) = {}'.format(res.x, res.fun))

        output_x = apply_model(model, res.x, shape, h0)
        print('Output = {}'.format(output_x))

        max_label = output_x.argmax(dim=1, keepdim=True)[0][0]

        if max_label == target:
            if best_dist > res.fun:
                best_x = res.x
                best_dist = res.fun
            cmax = c - 1
        else:
            cmin = c + 1

    print("Best distance = {}".format(best_dist))
    print("Best x = {}".format(best_x))

    output_x = apply_model(model, best_x, shape, h0)
    print('Output = {}'.format(output_x))


def func_robustness(x, shape, model, x0, target, distance, h0, c):
    if distance == 'll_0':
        loss1 = np.sum(x != x0)
    elif distance == 'll_2':
        loss1 = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        loss1 = np.max(np.absolute(x - x0))
    else:
        loss1 = 0

    output_x = apply_model(model, x, shape, h0)

    target_score = output_x[0][target]
    max_score = torch.max(output_x)

    loss2 = 0 if target_score >= max_score else max_score - target_score

    loss = loss1 + c * loss2

    return loss


def optimize_general(args, bnds, cons):
    shape = args[0] # shape
    model = args[1] # model

    x = args[2]     # x0, no need to copy
    h0 = args[-1]

    res = minimize(func_general, x, args=args, bounds=bnds, constraints=cons)

    print('Global minimum x0 = {}, f(x0) = {}'.format(res.x, res.fun))

    output_x = apply_model(model, res.x, shape, h0)
    print('Output = {}'.format(output_x))


def func_general(x, shape, model, x0, cons, h0):
    output_x = apply_model(model, x, shape, h0)

    loss = 0

    for con in cons:
        type = con['type']

        if type == 'max':
            i = con['index']
            m = torch.max(output_x).item()
            loss_i = 0 if output_x[0][i] == m else m - output_x[0][i]
        elif type == 'nmax':
            i = con['index']
            m = torch.max(output_x).item()
            loss_i = 0 if output_x[0][i] < m else 1
        elif type == 'min':
            i = con['index']
            m = torch.min(output_x).item()
            loss_i = 0 if output_x[0][i] == m else output_x[0][i] - m
        elif type == 'nmin':
            i = con['index']
            m = torch.min(output_x).item()
            loss_i = 0 if output_x[0][i] > m else 1
        else:
            size = len(output_x[0])
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
    torch.set_printoptions(threshold=20)
    np.set_printoptions(threshold=20)

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification input file')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    generate_adversarial_samples(spec)


if __name__ == '__main__':
    main()
