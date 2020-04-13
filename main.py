import argparse
import types
import json
import ast
import autograd.numpy as np
import torch
import lib

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from functools import partial


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
        x0 = np.maximum(x0, lb)
        x0 = np.minimum(x0, ub)

    h0 = None
    if 'h0' in spec:
        h0t = open(spec['h0'], 'r').readline()
        h0 = np.array(ast.literal_eval(h0t))

    if 'distance' in spec:
        distance = spec['distance']
    else:
        distance = 'll_i'

    print('Using', distance)

    cons = list()
    if 'fairness' in spec:
        for ind in ast.literal_eval(spec['fairness']):
            type = 'eq'
            val = x0[ind]
            fun = get_fairness(ind, val)
            cons.append({'type': type, 'fun': fun})

    print('x0 = {}'.format(x0.tolist()))

    output_x0 = apply_model(model, x0, shape, h0)
    print('Original output = {}'.format(output_x0))

    label_x0 = np.argmax(output_x0, axis=1)
    print('Original label = {}'.format(label_x0))

    # outfile = open('data.csv', 'w')
    #
    # s = str(label_x0)
    # s = s + ',' + ','.join(((x0 + 1) / 2 * 255).astype(int).astype(str).tolist()) + '\n'
    # outfile.write(s)

    if 'target' in spec:
        target = spec['target']
        args = (shape, model, x0, target, distance, h0)

        print('\nTarget = {}'.format(target))

        if target != label_x0:
            x = optimize_robustness(args, bnds, cons)
            x = post_process(x, spec)

            print('Final x = {}'.format(x.tolist()))

            # s = str(target)
            # s = s + ',' + ','.join((x * 255).astype(int).astype(str).tolist()) + '\n'
            # outfile.write(s)
    else:
        for i in range(spec['out_size']):
            target = i
            args = (shape, model, x0, target, distance, h0)

            print('\nTarget = {}'.format(target))

            if target != label_x0:
                x = optimize_robustness(args, bnds, cons)
                x = post_process(x, spec)

                print('Final x = {}'.format(x.tolist()))

                # s = str(target)
                # s = s + ',' + ','.join(((x + 1) / 2 * 255).astype(int).astype(str).tolist()) + '\n'
                # outfile.write(s)

    # outfile.flush()
    # outfile.close()


def generate_general(spec):
    size = spec['in_size']
    shape = ast.literal_eval(spec['in_shape'])

    lb, ub = get_bounds(spec)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    print('General linear constraints\n')

    x0 = np.zeros(size)

    h0 = None
    if 'h0' in spec:
        h0t = open(spec['h0'], 'r').readline()
        h0 = np.array(ast.literal_eval(h0t))

    print('x0 = {}'.format(x0.tolist()))
    output_x0 = apply_model(model, x0, shape, h0)
    print('Original output = {}\n'.format(output_x0))

    in_cons = list()

    if 'in_cons' in spec:
        for con in spec['in_cons']:
            type = con['type']
            coef = ast.literal_eval(con['coef'])
            fun = get_constraints(coef)
            in_cons.append({'type': type, 'fun': fun})

    out_cons = spec['out_cons']

    args = (shape, model, x0, out_cons, h0)
    x = optimize_general(args, bnds, in_cons)
    x = post_process(x, spec)

    print('Final x = {}'.format(x.tolist()))


def post_process(x, spec):
    if len(x) == 0:
        return x

    if 'rounding' in spec:
        print('Rounding x')
        for i in ast.literal_eval(spec['rounding']):
            x[i] = round(x[i])

    if 'one-hot' in spec:
        print('One-hot encoding x')
        rs = spec['one-hot']
        for r in rs:
            r = ast.literal_eval(r)
            amax = np.argmax(x[r[0]:r[1]]) + r[0]
            x[r[0]:r[1]] = 0
            x[amax] = 1

    return x


def get_model(spec):
    if 'model' in spec:
        model = torch.load(spec['model'])
    else:
        layers = spec['layers']
        ls = list()

        for layer in layers:
            if layer['type'] == 'linear':
                wt = open(layer['weight'], 'r').readline()
                bt = open(layer['bias'], 'r').readline()

                w = np.array(ast.literal_eval(wt))
                b = np.array(ast.literal_eval(bt))

                l = lib.Linear(w, b)

                ls.append(partial(l.apply))
            elif layer['type'] == 'function':
                f = layer['func']
                if f == 'relu':
                    ls.append(partial(np.maximum, 0))
                elif f == 'tanh':
                    ls.append(partial(np.tanh))

        model = generate_model(ls)

    return model


def generate_model(ls):
    def fun(x, ls=ls):
        res = x

        for l in ls:
            res = l(res)

        return res
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


def apply_model(model, x, shape, h):
    if isinstance(model, types.FunctionType):
        return apply_model_func(model, x, shape, h)
    else:
        return apply_model_pytorch(model, x, shape, h)


def apply_model_func(model, x, shape, h):
    if shape[0] == 1:
        x = x.reshape(shape)
        output_x = model(x)
    else:
        shape_x = [1,*shape[1:]]
        size_x = np.prod(shape[1:])

        shape_h = [1,h.size]
        h = h.reshape(shape_h)

        for i in range(shape[0]):
            xi = x[size_x * i : size_x * (i + 1)].reshape(shape_x)
            output_x, h = model(xi, h)

    return output_x


def apply_model_pytorch(model, x, shape, h):
    with torch.no_grad():
        if shape[0] == 1:
            x = torch.from_numpy(x).view(shape)
            output_x = model(x)
        else:
            x = torch.from_numpy(x)
            h = torch.from_numpy(h)

            shape_x = [1,*shape[1:]]
            size_x = np.prod(shape[1:])

            shape_h = [1,h.size()[0]]
            h = h.view(shape_h)

            for i in range(shape[0]):
                xi = x[size_x * i : size_x * (i + 1)].view(shape_x)
                output_x, h = model(xi, h)

    output_x = output_x.numpy()
    return output_x


def optimize_robustness(args, bnds, cons):
    shape = args[0]    # shape
    model = args[1]    # model

    target = args[3]   # target

    cmin = 1
    cmax = 100

    best_x = np.empty(0)
    best_dist = 1e9

    if isinstance(model, types.FunctionType):
        print('Model is a function. Jac is a function.')
        jac = grad(func_robustness)
    else:
        print('Model is a pytorch network. Jac is None.')
        jac = None

    while True:
        if cmin >= cmax: break

        c = int((cmin + cmax) / 2)
        print('c = {}'.format(c))

        x = args[2]    # x0
        h0 = args[-1]

        args_c = (*args, c)

        res = minimize(func_robustness, x, args=args_c, jac=jac, bounds=bnds, constraints=cons)

        print('Global minimum f(x) = {}'.format(res.fun))

        output_x = apply_model(model, res.x, shape, h0)
        print('Output = {}'.format(output_x))

        max_label = np.argmax(output_x, axis=1)

        if max_label == target:
            if best_dist > res.fun:
                best_x = res.x
                best_dist = res.fun
            cmax = c - 1
        else:
            cmin = c + 1

    print('Best distance = {}'.format(best_dist))

    if len(best_x) != 0:
        output_x = apply_model(model, best_x, shape, h0)
        print('Output = {}'.format(output_x))
    else:
        print('Failed to find x!')

    return best_x


def func_robustness(x, shape, model, x0, target, distance, h0, c):
    if distance == 'll_0':
        loss1 = np.sum(x != x0)
    elif distance == 'll_2':
        loss1 = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        loss1 = np.sum(np.abs(x - x0))
    else:
        loss1 = 0

    output_x = apply_model(model, x, shape, h0)
    target_score = output_x[0][target]

    output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
    max_score = np.max(output_x)

    loss2 = 0 if target_score > max_score else max_score - target_score + 1e-3

    loss = loss1 + c * loss2

    return loss


def optimize_general(args, bnds, cons):
    shape = args[0] # shape
    model = args[1] # model

    x = args[2]     # x0, no need to copy
    h0 = args[-1]

    res = minimize(func_general, x, args=args, bounds=bnds, constraints=cons)

    print('Global minimum f(x0) = {}'.format(res.fun))

    if res.fun <= 1e-6:
        print('SAT!')
    else:
        print('Unknown!')

    output_x = apply_model(model, res.x, shape, h0)
    print('Output = {}'.format(output_x))

    return res.x


def func_general(x, shape, model, x0, cons, h0):
    output_x = apply_model(model, x, shape, h0)

    loss = 0

    for con in cons:
        type = con['type']

        if type == 'max':
            i = con['index']
            m = np.max(output_x).item()
            loss_i = 0 if output_x[0][i] == m else m - output_x[0][i]
        elif type == 'nmax':
            i = con['index']
            m = np.max(output_x).item()
            loss_i = 0 if output_x[0][i] < m else 1
        elif type == 'min':
            i = con['index']
            m = np.min(output_x).item()
            loss_i = 0 if output_x[0][i] == m else output_x[0][i] - m
        elif type == 'nmin':
            i = con['index']
            m = np.min(output_x).item()
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
