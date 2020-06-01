import argparse
import types
import json
import ast
import autograd.numpy as np
import torch
import lib
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from functools import partial

def generate_adversarial_samples(spec, benchmark):
    if benchmark == 'mnist_challenge':
        run_mnist_challenge(spec)
    elif benchmark == 'eran_mnist':
        xpath = './benchmark/eran/data/mnist/'
        ypath = './benchmark/eran/data/labels/y_mnist.txt'
        run_benchmark(spec, xpath, ypath, 'mnist', False)
    elif benchmark == 'eran_mnist_norm':
        xpath = './benchmark/eran/data/mnist_norm/'
        ypath = './benchmark/eran/data/labels/y_mnist.txt'
        run_benchmark(spec, xpath, ypath, 'mnist', True)
    elif benchmark == 'eran_cifar':
        xpath = './benchmark/eran/data/cifar/'
        ypath = './benchmark/eran/data/labels/y_cifar.txt'
        run_benchmark(spec, xpath, ypath, 'cifar', False)
    elif benchmark == 'eran_cifar_norm':
        xpath = './benchmark/eran/data/cifar_norm/'
        ypath = './benchmark/eran/data/labels/y_cifar.txt'
        run_benchmark(spec, xpath, ypath, 'cifar', True)
    elif benchmark == 'fairness_bank':
        xpath = './benchmark/fairness/bank/data/'
        ypath = './benchmark/fairness/bank/data/labels.txt'
        run_benchmark(spec, xpath, ypath)
    elif benchmark == 'jigsaw_gru' or benchmark == 'jigsaw_lstm':
        xpath = './benchmark/rnn/data/jigsaw/'
        ypath = './benchmark/rnn/data/jigsaw/labels.txt'
        run_benchmark(spec, xpath, ypath, 'jigsaw')
    elif benchmark == 'wiki_gru' or benchmark == 'wiki_lstm':
        xpath = './benchmark/rnn/data/wiki/'
        ypath = './benchmark/rnn/data/wiki/labels.txt'
        run_benchmark(spec, xpath, ypath, 'wiki')
    elif benchmark == 'fairness_census':
        xpath = './benchmark/fairness/census/data/'
        ypath = './benchmark/fairness/census/data/labels.txt'
        run_benchmark(spec, xpath, ypath)
    elif benchmark == 'fairness_credit':
        xpath = './benchmark/fairness/credit/data/'
        ypath = './benchmark/fairness/credit/data/labels.txt'
        run_benchmark(spec, xpath, ypath)
    elif 'robustness' in spec:
        generate_robustness(spec)
    else:
        generate_general(spec)


def run_benchmark(spec, xpath, ypath, dataset=None, is_norm=None):
    if 'solver' in spec and spec['solver'] == 'sprt':
        solver = 'sprt'
    else:
        solver = 'optimize'

    if dataset != 'jigsaw' and dataset != 'wiki':
        shape = ast.literal_eval(spec['shape'])
        size = np.prod(shape)

        lb, ub = get_bounds(spec, size)
        bnds = Bounds(lb, ub)
        model = get_model(spec)

    if 'distance' in spec:
        distance = spec['distance']
    else:
        distance = 'll_i'

    print('Using', distance)

    eps = 0.5
    print('eps = {}'.format(eps))

    yfile = open(ypath, 'r')
    ytext = yfile.readline()
    y0s = np.array(ast.literal_eval(ytext))

    batch = 100
    if dataset == 'jigsaw' or dataset == 'wiki':
        batch = 10

    for i in range(batch):
        xfile = open(xpath + 'data' + str(i) + '.txt', 'r')
        xtext = xfile.readline()
        x0 = np.array(ast.literal_eval(xtext))

        if dataset == 'jigsaw' or dataset == 'wiki':
            size = len(x0)
            shape = [int(size / 50), 50]

            bnds = spec['bounds']
            bnds = ast.literal_eval(bnds)

            lb = np.full(size, bnds[0])
            ub = np.full(size, bnds[1])

            if solver == 'sprt':
                x0_l = x0 - eps
                x0_u = x0 + eps

                lb = np.maximum(lb, x0_l)
                ub = np.minimum(ub, x0_u)

                bnds1 = Bounds(lb, ub)
            else:
                bnds = Bounds(lb, ub)

            model = get_model(spec, shape)

        if solver == 'sprt' and (dataset == 'mnist' or dataset == 'cifar'):
            lb = bnds.lb
            ub = bnds.ub

            x0_d = bench_denormalize(x0, dataset, is_norm)

            x0_l = x0_d - eps
            x0_u = x0_d + eps

            x0_l = bench_normalize(x0_l, dataset, is_norm)
            x0_u = bench_normalize(x0_u, dataset, is_norm)

            lb = np.maximum(lb, x0_l)
            ub = np.minimum(ub, x0_u)

            bnds1 = Bounds(lb, ub)

        print('\n=================================\n')
        print('x0 = {}'.format(x0))

        output_x0 = apply_model(model, x0, shape)
        print('Original output = {}'.format(output_x0))

        label_x0 = np.argmax(output_x0, axis=1)
        print('Original label = {}'.format(label_x0))

        cons = list()
        if 'fairness' in spec:
            if solver == 'sprt':
                lb = bnds.lb
                ub = bnds.ub

                for ind in ast.literal_eval(spec['fairness']):
                    lb[ind] = x0[ind]
                    ub[ind] = x0[ind]

                bnds1 = Bounds(lb, ub)
            else:
                for ind in ast.literal_eval(spec['fairness']):
                    type = 'eq'
                    val = x0[ind]
                    fun = get_fairness(ind, val)
                    cons.append({'type': type, 'fun': fun})

        target = y0s[i]

        args = (shape, model, x0, target, distance, False)

        print('\nUntarget = {}'.format(target))

        if target == label_x0:
            if solver == 'sprt':
                confidence = ast.literal_eval(spec['confidence'])
                alpha = ast.literal_eval(spec['alpha'])
                beta = ast.literal_eval(spec['beta'])
                gamma = ast.literal_eval(spec['gamma'])

                params = (confidence, alpha, beta, gamma)

                x = sprt(args, bnds1, cons, params)
            else:
                x = optimize_robustness(args, bnds, cons)

            if len(x) != 0:
                print('Final x = {}'.format(x))

                x0 = bench_denormalize(x0, dataset, is_norm)
                x = bench_denormalize(x, dataset, is_norm)

                d = final_distance(distance, x, x0)
                print('Final distance = {}'.format(d))

                output_x = apply_model(model, x, shape)
                print('Final output = {}'.format(output_x))
            else:
                print('Failed to find x!')


def bench_denormalize(x, dataset, is_norm):
    if dataset == 'mnist':
        if is_norm:
            x = x * 0.3081 + 0.1307
        else:
            x = x
    elif dataset == 'cifar':
        if is_norm:
            x = x.copy()
            x[0:1024] = x[0:1024] * 0.2023 + 0.4914
            x[1024:2048] = x[1024:2048] * 0.1994 + 0.4822
            x[2048:3072] = x[2048:3072] * 0.2010 + 0.4465
        else:
            x = x + 0.5

    return x


def bench_normalize(x, dataset, is_norm):
    if dataset == 'mnist':
        if is_norm:
            x = (x - 0.1307) / 0.3081
        else:
            x = x
    elif dataset == 'cifar':
        if is_norm:
            x = x.copy()
            x[0:1024] = (x[0:1024] - 0.4914) / 0.2023
            x[1024:2048] = (x[1024:2048] - 0.4822) / 0.1994
            x[2048:3072] = (x[2048:3072] - 0.4465) / 0.2010
        else:
            x = x - 0.5

    return x


def run_mnist_challenge(spec):
    count = 0

    shape = ast.literal_eval(spec['shape'])
    size = np.prod(shape)

    model = get_model(spec)

    distance = 'll_i'
    print('Using', distance)

    eps = -1.0

    cons = list()

    for i in range(1):
        xfile = open('benchmark/mnist_challenge/x&y/x' + str(i) + '.txt', 'r')
        yfile = open('benchmark/mnist_challenge/x&y/y' + str(i) + '.txt', 'r')

        xtext = xfile.readline()
        ytext = yfile.readline()

        xdata = np.array(ast.literal_eval(xtext))
        ydata = np.array(ast.literal_eval(ytext))

        for j in range(10):
            x0 = xdata[j]

            lb = x0 - 0.3
            lb = np.maximum(lb, 0)

            ub = x0 + 0.3
            ub = np.minimum(ub, 1)

            bnds = Bounds(lb, ub)

            print('\n=================================\n')
            print('x0 = {}'.format(x0))

            output_x0 = apply_model(model, x0, shape)
            print('Original output = {}'.format(output_x0))

            label_x0 = np.argmax(output_x0, axis=1)
            print('Original label = {}'.format(label_x0))

            target = ydata[j]

            args = (shape, model, x0, target, distance, False)

            print('\nUntarget = {}'.format(target))

            if target == label_x0:
                if 'solver' in spec and spec['solver'] == 'sprt':
                    confidence = ast.literal_eval(spec['confidence'])
                    alpha = ast.literal_eval(spec['alpha'])
                    beta = ast.literal_eval(spec['beta'])
                    gamma = ast.literal_eval(spec['gamma'])

                    params = (confidence, alpha, beta, gamma)

                    x = sprt(args, bnds, cons, params)
                else:
                    x = optimize_robustness(args, bnds, cons)

                if len(x) != 0:
                    print('Final x = {}'.format(x))

                    d = final_distance(distance, x, x0)
                    print('Final distance = {}'.format(d))

                    if d <= 0.3:
                        count = count + 1

                    output_x = apply_model(model, x, shape)
                    print('Final output = {}'.format(output_x))
                else:
                    print('Failed to find x!')

    print('Result = {}/10000'.format(count))


def display(x0, y0, x, y, shape):
    fig, ax = plt.subplots(1, 2)

    x0 = x0 * 255
    x0 = x0.astype('uint8')
    x0 = x0.reshape(shape)

    ax[0].set(title='Original. Label is {}'.format(y0))
    if len(shape) == 2:
        ax[0].imshow(x0, cmap='gray')
    elif len(shape) == 3:
        x0 = x0.transpose(1, 2, 0)
        ax[0].imshow(x0)

    x = x * 255
    x = x.astype('uint8')
    x = x.reshape(shape)

    ax[1].set(title='Adv. sample. Label is {}'.format(y))
    if len(shape) == 2:
        ax[1].imshow(x, cmap='gray')
    elif len(shape) == 3:
        x = x.transpose(1, 2, 0)
        ax[1].imshow(x)

    plt.show()


def denormalize(x, mean=0, std=1):
    x = x * std + mean

    return x


def final_distance(distance, x, x0):
    d = 1e9

    if distance == 'll_0':
        d = np.sum(x != x0)
    elif distance == 'll_2':
        d = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        d = np.max(np.abs(x - x0))

    return d


def generate_robustness(spec):
    shape = ast.literal_eval(spec['shape'])
    size = np.prod(shape)

    lb, ub = get_bounds(spec, size)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    if spec['robustness'] == 'local':
        print('Local robustness\n')
        max_iter = 1
    else:
        print('Global robustness\n')
        max_iter = 100

    if 'distance' in spec:
        distance = spec['distance']
    else:
        distance = 'll_i'

    print('Using', distance)

    if 'eps' in spec:
        eps = ast.literal_eval(spec['eps'])
    else:
        eps = -1.0

    cons = list()
    if 'fairness' in spec:
        for ind in ast.literal_eval(spec['fairness']):
            type = 'eq'
            val = x0[ind]
            fun = get_fairness(ind, val)
            cons.append({'type': type, 'fun': fun})

    def run():
        if 'solver' in spec and spec['solver'] == 'sprt':
            confidence = ast.literal_eval(spec['confidence'])
            alpha = ast.literal_eval(spec['alpha'])
            beta = ast.literal_eval(spec['beta'])
            gamma = ast.literal_eval(spec['gamma'])

            params = (confidence, alpha, beta, gamma)

            x = sprt(args, bnds, cons, params)
        else:
            x = optimize_robustness(args, bnds, cons)

        x = post_process(x, spec)

        if len(x) != 0:
            print('Final x = {}'.format(x))

            d = final_distance(distance, x, x0)
            print('Final distance = {}'.format(d))

            output_x = apply_model(model, x, shape)
            print('Final output = {}'.format(output_x))

            label_x = np.argmax(output_x, axis=1)
            print('Final label = {}'.format(label_x))

            if 'display' in spec and spec['display'] == 'on':
                if 'dmean' in spec:
                    dmean = ast.literal_eval(spec['dmean'])
                else:
                    dmean = 0

                if 'dstd' in spec:
                    dstd = ast.literal_eval(spec['dstd'])
                else:
                    dstd = 1

                x0_denorm = denormalize(x0, dmean, dstd)
                x_denorm = denormalize(x, dmean, dstd)

                dshape = ast.literal_eval(spec['dshape'])

                display(x0_denorm, label_x0, x_denorm, label_x, dshape)

            return True
        else:
            print('Failed to find x!')
            return False

    for i in range(max_iter):
        if spec['robustness'] == 'local':
            x0t = open(spec['x0'], 'r').readline()
            x0 = np.array(ast.literal_eval(x0t))
        else:
            x0 = np.random.rand(size) * 2 - 1
            x0 = np.maximum(x0, lb)
            x0 = np.minimum(x0, ub)

        print('x0 = {}'.format(x0))

        output_x0 = apply_model(model, x0, shape)
        print('Original output = {}'.format(output_x0))

        label_x0 = np.argmax(output_x0, axis=1)
        print('Original label = {}'.format(label_x0))

        if 'target' in spec:
            target = spec['target']
            args = (shape, model, x0, target, distance, True)

            print('\nTarget = {}'.format(target))

            if target != label_x0:
                res = run()
                if res: break
        elif 'untarget' in spec:
            target = spec['untarget']

            args = (shape, model, x0, target, distance, False)

            print('\nUntarget = {}'.format(target))

            if target == label_x0:
                res = run()
                if res: break


def generate_general(spec):
    shape = ast.literal_eval(spec['shape'])
    size = np.prod(shape)

    lb, ub = get_bounds(spec, size)
    bnds = Bounds(lb, ub)
    model = get_model(spec)

    print('General linear constraints\n')

    x0 = np.zeros(size)
    print('x0 = {}'.format(x0))

    output_x0 = apply_model(model, x0, shape)
    print('Original output = {}\n'.format(output_x0))

    in_cons = list()

    if 'in_cons' in spec:
        for con in spec['in_cons']:
            type = con['type']
            coef = ast.literal_eval(con['coef'])
            fun = get_constraints(coef)
            in_cons.append({'type': type, 'fun': fun})

    out_cons = spec['out_cons']

    args = (shape, model, x0, out_cons)

    if 'solver' in spec and spec['solver'] == 'sprt':
        confidence = ast.literal_eval(spec['confidence'])
        alpha = ast.literal_eval(spec['alpha'])
        beta = ast.literal_eval(spec['beta'])
        gamma = ast.literal_eval(spec['gamma'])

        params = (confidence, alpha, beta, gamma)

        x = sprt(args, bnds, in_cons, params)
    else:
        x = optimize_general(args, bnds, in_cons)

    x = post_process(x, spec)

    if len(x) != 0:
        print('Final x = {}'.format(x))

        output_x = apply_model(model, x, shape)
        print('Final output = {}'.format(output_x))
    else:
        print('Failed to find x!')


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


def get_model(spec, shape=None):
    if 'model' in spec:
        model = torch.load(spec['model'])
    else:
        if shape == None:
            shape = ast.literal_eval(spec['shape'])
        len = shape[0]

        layers = spec['layers']
        ls = list()

        def add_func(layer, ls):
            if 'func' in layer:
                f = layer['func']

                if f == 'relu':
                    ls.append(partial(lib.relu))
                elif f == 'sigmoid':
                    ls.append(partial(lib.sigmoid))
                elif f == 'tanh':
                    ls.append(partial(np.tanh))
                elif f == 'reshape':
                    ns = ast.literal_eval(layer['newshape'])
                    import numpy as rnp
                    ls.append(partial(rnp.reshape, newshape=ns))
                elif f == 'transpose':
                    ax = ast.literal_eval(layer['axes'])
                    import numpy as rnp
                    ls.append(partial(rnp.transpose, axes=ax))
                elif f != 'softmax':
                    raise NameError('Not support yet!')

        for layer in layers:
            if layer['type'] == 'linear':
                wt = open(layer['weights'], 'r').readline()
                bt = open(layer['bias'], 'r').readline()

                weights = np.array(ast.literal_eval(wt))
                bias = np.array(ast.literal_eval(bt))

                l = lib.Linear(weights, bias)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'conv1d' or layer['type'] == 'conv2d' \
                or layer['type'] == 'conv3d':
                ft = open(layer['filters'], 'r').readline()
                bt = open(layer['bias'], 'r').readline()

                filters = np.array(ast.literal_eval(ft))
                bias = np.array(ast.literal_eval(bt))

                stride = layer['stride']
                padding = layer['padding']

                if layer['type'] == 'conv1d':
                    l = lib.Conv1d(filters, bias, stride, padding)
                elif layer['type'] == 'conv2d':
                    l = lib.Conv2d(filters, bias, stride, padding)
                elif layer['type'] == 'conv3d':
                    l = lib.Conv3d(filters, bias, stride, padding)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'maxpool1d' or layer['type'] == 'maxpool2d' \
                or layer['type'] == 'maxpool3d':
                kt = layer['kernel']

                kernel = np.array(ast.literal_eval(kt))

                stride = layer['stride']
                padding = layer['padding']

                if layer['type'] == 'maxpool1d':
                    l = lib.MaxPool1d(kernel, stride, padding)
                elif layer['type'] == 'maxpool2d':
                    l = lib.MaxPool2d(kernel, stride, padding)
                elif layer['type'] == 'maxpool3d':
                    l = lib.MaxPool3d(kernel, stride, padding)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'resnet2l':
                f1t = open(layer['filters1'], 'r').readline()
                b1t = open(layer['bias1'], 'r').readline()
                f2t = open(layer['filters2'], 'r').readline()
                b2t = open(layer['bias2'], 'r').readline()

                filters1 = np.array(ast.literal_eval(f1t))
                bias1 = np.array(ast.literal_eval(b1t))
                filters2 = np.array(ast.literal_eval(f2t))
                bias2 = np.array(ast.literal_eval(b2t))

                stride1 = layer['stride1']
                padding1 = layer['padding1']
                stride2 = layer['stride2']
                padding2 = layer['padding2']

                if 'filterX' in layer:
                    fXt = open(layer['filtersX'], 'r').readline()
                    bXt = open(layer['biasX'], 'r').readline()

                    filtersX = np.array(ast.literal_eval(fXt))
                    biasX = np.array(ast.literal_eval(bXt))

                    strideX = layer['strideX']
                    paddingX = layer['paddingX']

                    l = lib.ResNet2l(filters1, bias1, stride1, padding1,
                        filters2, bias2, stride2, padding2,
                        filtersX, biasX, strideX, paddingX)
                else:
                    l = lib.ResNet2l(filters1, bias1, stride1, padding1,
                        filters2, bias2, stride2, padding2)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'resnet3l':
                f1t = open(layer['filters1'], 'r').readline()
                b1t = open(layer['bias1'], 'r').readline()
                f2t = open(layer['filters2'], 'r').readline()
                b2t = open(layer['bias2'], 'r').readline()
                f3t = open(layer['filters3'], 'r').readline()
                b3t = open(layer['bias3'], 'r').readline()

                filters1 = np.array(ast.literal_eval(f1t))
                bias1 = np.array(ast.literal_eval(b1t))
                filters2 = np.array(ast.literal_eval(f2t))
                bias2 = np.array(ast.literal_eval(b2t))
                filters3 = np.array(ast.literal_eval(f3t))
                bias3 = np.array(ast.literal_eval(b3t))

                stride1 = layer['stride1']
                padding1 = layer['padding1']
                stride2 = layer['stride2']
                padding2 = layer['padding2']
                stride3 = layer['stride3']
                padding3 = layer['padding3']

                if 'filterX' in layer:
                    fXt = open(layer['filtersX'], 'r').readline()
                    bXt = open(layer['biasX'], 'r').readline()

                    filtersX = np.array(ast.literal_eval(fXt))
                    biasX = np.array(ast.literal_eval(bXt))

                    strideX = layer['strideX']
                    paddingX = layer['paddingX']

                    l = lib.ResNet3l(filters1, bias1, stride1, padding1,
                        filters2, bias2, stride2, padding2,
                        filters3, bias3, stride3, padding3,
                        filtersX, biasX, strideX, paddingX)
                else:
                    l = lib.ResNet3l(filters1, bias1, stride1, padding1,
                        filters2, bias2, stride2, padding2,
                        filters3, bias3, stride3, padding3)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'relurnn' or layer['type'] == 'tanhrnn':
                wiht = open(layer['weights_ih'], 'r').readline()
                whht = open(layer['weights_hh'], 'r').readline()
                biht = open(layer['bias_ih'], 'r').readline()
                bhht = open(layer['bias_hh'], 'r').readline()
                h0t = open(layer['h0'], 'r').readline()

                weights_ih = np.array(ast.literal_eval(wiht))
                weights_hh = np.array(ast.literal_eval(whht))
                bias_ih = np.array(ast.literal_eval(biht))
                bias_hh = np.array(ast.literal_eval(bhht))
                h0 = np.array(ast.literal_eval(h0t))

                if layer['type'] == 'relurnn':
                    l = lib.ReluRNN(weights_ih, weights_hh, bias_ih, bias_hh, h0)
                elif layer['type'] == 'tanhrnn':
                    l = lib.TanhRNN(weights_ih, weights_hh, bias_ih, bias_hh, h0)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'lstm':
                wt = open(layer['weights'], 'r').readline()
                bt = open(layer['bias'], 'r').readline()

                h0t = open(layer['h0'], 'r').readline()
                c0t = open(layer['c0'], 'r').readline()

                weights = np.array(ast.literal_eval(wt))
                bias = np.array(ast.literal_eval(bt))

                h0 = np.array(ast.literal_eval(h0t))
                c0 = np.array(ast.literal_eval(c0t))

                l = lib.LSTM(weights, bias, h0, c0, len)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'gru':
                gwt = open(layer['gate_weights'], 'r').readline()
                cwt = open(layer['candidate_weights'], 'r').readline()

                gbt = open(layer['gate_bias'], 'r').readline()
                cbt = open(layer['candidate_bias'], 'r').readline()

                h0t = open(layer['h0'], 'r').readline()

                gate_weights = np.array(ast.literal_eval(gwt))
                candidate_weights = np.array(ast.literal_eval(cwt))

                gate_bias = np.array(ast.literal_eval(gbt))
                candidate_bias = np.array(ast.literal_eval(cbt))

                h0 = np.array(ast.literal_eval(h0t))

                l = lib.GRU(gate_weights, candidate_weights, \
                    gate_bias, candidate_bias, h0, len)

                ls.append(partial(l.apply))

                add_func(layer, ls)
            elif layer['type'] == 'function':
                add_func(layer, ls)

        model = generate_model(ls)

    return model


def generate_model(ls):
    def fun(x, ls=ls):
        res = x
        i = 0

        for l in ls:
            res = l(res)

        return res
    return fun


def get_bounds(spec, size):
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


def apply_model(model, x, shape):
    if isinstance(model, types.FunctionType):
        return apply_model_func(model, x, shape)
    else:
        return apply_model_pytorch(model, x, shape)


def apply_model_func(model, x, shape):
    if shape[0] == 1:
        x = x.reshape(shape)
        output_x = model(x)
    else:
        shape_x = [1,*shape[1:]]
        size_x = np.prod(shape[1:])

        for i in range(shape[0]):
            xi = x[size_x * i : size_x * (i + 1)].reshape(shape_x)
            output_x = model(xi)

    return output_x


def apply_model_pytorch(model, x, shape):
    with torch.no_grad():
        if shape[0] == 1:
            x = torch.from_numpy(x).view(shape)
            output_x = model(x)
        else:
            x = torch.from_numpy(x)

            shape_x = [1,*shape[1:]]
            size_x = np.prod(shape[1:])

            for i in range(shape[0]):
                xi = x[size_x * i : size_x * (i + 1)].view(shape_x)
                output_x = model(xi)

    output_x = output_x.numpy()
    return output_x


def sprt_generate_x(shape, bnds):
    size = np.prod(shape)
    x = np.random.rand(size)

    lb = bnds.lb
    ub = bnds.ub

    x = (ub - lb) * x + lb

    return x


def sprt_is_sat_in_cons(cons, x):
    for con in cons:
        type = con['type']
        size = len(x)
        coef = ast.literal_eval(con['coef'])

        sum = 0
        for i in range(size):
            sum += coef[i] * x[i]
        sum +=coef[size]

        if type == 'eq' and sum != 0: return False
        elif type == 'ineq' and sum < 0: return False

    return True


def sprt_is_sat_robustness(shape, model, x0, target, distance, is_targeted, x):
    # print('x = {}'.format(x.tolist()))

    output_x = apply_model(model, x, shape)

    # print('output_x = {}'.format(output_x))

    max_label = np.argmax(output_x, axis=1)

    # print('max_label = {}'.format(max_label))

    if is_targeted:
        if max_label == target:
            return True
        else:
            return False
    else:
        if max_label != target:
            return True
        else:
            return False


def sprt_is_sat_general(shape, model, x0, out_cons, x):
    # remember that the out_cons is the negation of desired properties
    # we are checking sat of out_cons
    output_x = apply_model(model, x, shape)

    target1 = 3
    target2 = 4

    target_score1 = output_x[0][target1]
    target_score2 = output_x[0][target2]

    output_x = output_x - np.eye(output_x[0].size)[target1] * 1e6 - np.eye(output_x[0].size)[target2] * 1e6
    min_score = np.min(output_x)

    if target_score1 > min_score and target_score2 > min_score: return False

    # for con in out_cons:
    #     type = con['type']
    #
    #     if type == 'max':
    #         target = con['index']
    #         target_score = output_x[0][target]
    #
    #         output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
    #         max_score = np.max(output_x)
    #
    #         if target_score < max_score: return False
    #     elif type == 'nmax':
    #         target = con['index']
    #         target_score = output_x[0][target]
    #
    #         output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
    #         max_score = np.max(output_x)
    #
    #         if target_score > max_score: return False
    #     elif type == 'min':
    #         target = con['index']
    #         target_score = output_x[0][target]
    #
    #         output_x = output_x + np.eye(output_x[0].size)[target] * 1e6
    #         min_score = np.min(output_x)
    #
    #         if target_score > min_score: return False
    #     elif type == 'nmin':
    #         target = con['index']
    #         target_score = output_x[0][target]
    #
    #         output_x = output_x + np.eye(output_x[0].size)[target] * 1e6
    #         min_score = np.min(output_x)
    #
    #         if target_score < min_score: return False
    #     else:
    #         size = len(output_x[0])
    #         coef = ast.literal_eval(con['coef'])
    #
    #         sum = 0
    #         for i in range(size):
    #             sum += coef[i] * output_x[0][i]
    #         sum +=coef[size]
    #
    #         if type == 'eq' and sum != 0: return False
    #         elif type == 'ineq' and sum < 0: return False

    return True


def sprt_is_sat(args, cons, x):
    if sprt_is_sat_in_cons(cons, x):
        if len(args) == 6:
            return sprt_is_sat_robustness(*args, x)
        elif len(args) == 4:
            return sprt_is_sat_general(*args, x)
    else:
        return False


def sprt(args, bnds, cons, params):
    confidence, alpha, beta, gamma = params

    p0 = confidence + gamma
    p1 = confidence - gamma

    lower = beta / (1 - alpha)
    upper = (1 - beta) / alpha

    pr = 1
    no_x = 0

    while True:
        x = sprt_generate_x(args[0], bnds)
        no_x = no_x + 1

        sat = sprt_is_sat(args, cons, x)

        if sat:
            pr = pr * (1 - p1) / (1 - p0)
        else:
            pr = pr * p1 / p0

        if pr <= lower:
            print('Accept H0. The formula is UNSAT with p >= {} after {} tests.'.format(p0, no_x))
            break
        elif pr >= upper:
            print('Accept H1. The formula is UNSAT with p <= {} after {} tests.'.format(p1, no_x))
            break

    return np.empty(0)


def optimize_robustness(args, bnds, cons):
    shape = args[0]    # shape
    model = args[1]    # model

    target = args[3]   # target
    is_targeted = args[-1]  # target or untarget

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

        x = args[2]    # x0

        args_c = (*args, c)

        res = minimize(func_robustness, x, args=args_c, jac=jac, bounds=bnds, constraints=cons)

        output_x = apply_model(model, res.x, shape)

        max_label = np.argmax(output_x, axis=1)

        if is_targeted:
            if max_label == target:
                if best_dist > res.fun:
                    best_x = res.x
                    best_dist = res.fun
                cmax = c - 1
            else:
                cmin = c + 1
        else:
            if max_label != target:
                if best_dist > res.fun:
                    best_x = res.x
                    best_dist = res.fun
                cmax = c - 1
            else:
                cmin = c + 1

    return best_x


def func_robustness(x, shape, model, x0, target, distance, is_targeted, c):
    if distance == 'll_0':
        loss1 = np.sum(x != x0)
    elif distance == 'll_2':
        loss1 = np.sqrt(np.sum((x - x0) ** 2))
    elif distance == 'll_i':
        loss1 = np.max(np.abs(x - x0))
    else:
        loss1 = 0

    output_x = apply_model(model, x, shape)
    target_score = output_x[0][target]

    output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
    max_score = np.max(output_x)

    if is_targeted:
        loss2 = 0 if target_score > max_score else max_score - target_score + 1e-3
    else:
        loss2 = 0 if target_score < max_score else target_score - max_score + 1e-3

    loss = loss1 + c * loss2

    return loss


def optimize_general(args, bnds, cons):
    shape = args[0] # shape
    model = args[1] # model

    x = args[2]     # x0, no need to copy

    res = minimize(func_general, x, args=args, bounds=bnds, constraints=cons)

    if res.fun <= 1e-6:
        print('SAT!')
    else:
        print('Unknown!')

    return res.x


def func_general(x, shape, model, x0, cons):
    output_x = apply_model(model, x, shape)

    loss = 0

    for con in cons:
        type = con['type']

        if type == 'max':
            target = con['index']
            target_score = output_x[0][target]

            output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
            max_score = np.max(output_x)

            loss_i = 0 if target_score > max_score else max_score - target_score + 1e-3
        elif type == 'nmax':
            target = con['index']
            target_score = output_x[0][target]

            output_x = output_x - np.eye(output_x[0].size)[target] * 1e6
            max_score = np.max(output_x)

            loss_i = 0 if target_score < max_score else target_score - max_score + 1e-3
        elif type == 'min':
            target = con['index']
            target_score = output_x[0][target]

            output_x = output_x + np.eye(output_x[0].size)[target] * 1e6
            min_score = np.min(output_x)

            loss_i = 0 if target_score < min_score else target_score - min_score + 1e-3
        elif type == 'nmin':
            target = con['index']
            target_score = output_x[0][target]

            output_x = output_x + np.eye(output_x[0].size)[target] * 1e6
            min_score = np.min(output_x)

            loss_i = 0 if target_score > min_score else min_score - target_score + 1e-3
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
                loss_i = 0 if sum >= 0 else abs(sum)

        loss += loss_i

    return loss


def main():
    torch.set_printoptions(threshold=20)
    np.set_printoptions(threshold=20)

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification input file')
    parser.add_argument('--bench', type=str, default='None',
                        help='run different benchmark')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    generate_adversarial_samples(spec, args.bench)


if __name__ == '__main__':
    main()
