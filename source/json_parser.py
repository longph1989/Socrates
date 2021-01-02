import autograd.numpy as np
import ast

from antlr4 import *
from assertion.AssertionLexer import AssertionLexer
from assertion.AssertionParser import AssertionParser
from assertion.AssertionVisitor import AssertionVisitor

from model.lib_models import *
from model.lib_layers import *
from assertion.lib_functions import set_model
from solver.lib_solvers import *
from utils import *
from display import *


def parse_layers(spec):
    layers = list()

    for layer in spec:
        type = layer['type'].lower()

        if type == 'linear':

            weights = np.array(ast.literal_eval(read(layer['weights'])))
            bias = np.array(ast.literal_eval(read(layer['bias'])))
            name = layer['func'] if 'func' in layer else None

            layers.append(Linear(weights, bias, name))

        elif type == 'conv1d' or type == 'conv2d' \
            or type == 'conv3d':

            filters = np.array(ast.literal_eval(read(layer['filters'])))
            bias = np.array(ast.literal_eval(read(layer['bias'])))

            stride = ast.literal_eval(read(layer['stride']))
            padding = ast.literal_eval(read(layer['padding']))

            if type == 'conv1d':
                layers.append(Conv1d(filters, bias, stride, padding))
            elif type == 'conv2d':
                layers.append(Conv2d(filters, bias, stride, padding))
            elif type == 'conv3d':
                layers.append(Conv3d(filters, bias, stride, padding))

        elif type == 'maxpool1d' or type == 'maxpool2d' \
            or type == 'maxpool3d':

            kernel = np.array(ast.literal_eval(read(layer['kernel'])))

            stride = ast.literal_eval(read(layer['stride']))
            padding = ast.literal_eval(read(layer['padding']))

            if type == 'maxpool1d':
                layers.append(MaxPool1d(kernel, stride, padding))
            elif type == 'maxpool2d':
                layers.append(MaxPool2d(kernel, stride, padding))
            elif type == 'maxpool3d':
                layers.append(MaxPool3d(kernel, stride, padding))

        elif type == 'resnet2l':

            filters1 = np.array(ast.literal_eval(read(layer['filters1'])))
            bias1 = np.array(ast.literal_eval(read(layer['bias1'])))
            filters2 = np.array(ast.literal_eval(read(layer['filters2'])))
            bias2 = np.array(ast.literal_eval(read(layer['bias2'])))

            stride1 = ast.literal_eval(read(layer['stride1']))
            padding1 = ast.literal_eval(read(layer['padding1']))
            stride2 = ast.literal_eval(read(layer['stride2']))
            padding2 = ast.literal_eval(read(layer['padding2']))

            if 'filterX' in layer:
                filtersX = np.array(ast.literal_eval(read(layer['filtersX'])))
                biasX = np.array(ast.literal_eval(read(layer['biasX'])))

                strideX = ast.literal_eval(read(layer['strideX']))
                paddingX = ast.literal_eval(read(layer['paddingX']))

                layers.append(ResNet2l(filters1, bias1, stride1, padding1,
                    filters2, bias2, stride2, padding2,
                    filtersX, biasX, strideX, paddingX))
            else:
                layers.append(ResNet2l(filters1, bias1, stride1, padding1,
                    filters2, bias2, stride2, padding2))

        elif type == 'resnet3l':

            filters1 = np.array(ast.literal_eval(read(layer['filters1'])))
            bias1 = np.array(ast.literal_eval(read(layer['bias1'])))
            filters2 = np.array(ast.literal_eval(read(layer['filters2'])))
            bias2 = np.array(ast.literal_eval(read(layer['bias2'])))
            filters3 = np.array(ast.literal_eval(read(layer['filters3'])))
            bias3 = np.array(ast.literal_eval(read(layer['bias3'])))

            stride1 = ast.literal_eval(read(layer['stride1']))
            padding1 = ast.literal_eval(read(layer['padding1']))
            stride2 = ast.literal_eval(read(layer['stride2']))
            padding2 = ast.literal_eval(read(layer['padding2']))
            stride3 = ast.literal_eval(read(layer['stride3']))
            padding3 = ast.literal_eval(read(layer['padding3']))

            if 'filterX' in layer:
                filtersX = np.array(ast.literal_eval(read(layer['filtersX'])))
                biasX = np.array(ast.literal_eval(read(layer['biasX'])))

                strideX = ast.literal_eval(read(layer['strideX']))
                paddingX = ast.literal_eval(read(layer['paddingX']))

                layers.append(ResNet3l(filters1, bias1, stride1, padding1,
                    filters2, bias2, stride2, padding2,
                    filters3, bias3, stride3, padding3,
                    filtersX, biasX, strideX, paddingX))
            else:
                layers.append(ResNet3l(filters1, bias1, stride1, padding1,
                    filters2, bias2, stride2, padding2,
                    filters3, bias3, stride3, padding3))

        elif type == 'rnn':

            weights = np.array(ast.literal_eval(read(layer['weights'])))
            bias = np.array(ast.literal_eval(read(layer['bias'])))
            h0 = np.array(ast.literal_eval(read(layer['h0'])))
            name = layer['func'] if 'func' in layer else None

            layers.append(BasicRNN(weights, bias, h0, name))

        elif type == 'lstm':

            weights = np.array(ast.literal_eval(read(layer['weights'])))
            bias = np.array(ast.literal_eval(read(layer['bias'])))

            h0 = np.array(ast.literal_eval(read(layer['h0'])))
            c0 = np.array(ast.literal_eval(read(layer['c0'])))

            layers.append(LSTM(weights, bias, h0, c0))

        elif type == 'gru':

            gate_weights = np.array(ast.literal_eval(read(layer['gate_weights'])))
            candidate_weights = np.array(ast.literal_eval(read(layer['candidate_weights'])))

            gate_bias = np.array(ast.literal_eval(read(layer['gate_bias'])))
            candidate_bias = np.array(ast.literal_eval(read(layer['candidate_bias'])))

            h0 = np.array(ast.literal_eval(read(layer['h0'])))

            layers.append(GRU(gate_weights, candidate_weights, gate_bias, candidate_bias, h0))

        elif type == 'function':
            name = layer['func'].lower()

            if name == 'reshape':
                ns = ast.literal_eval(read(layer['newshape']))
                params = (ns)
            elif name == 'transpose':
                ax = ast.literal_eval(read(layer['axes']))
                params = (ax)
            else:
                params = None

            layers.append(Function(name, params))

    return layers


def parse_bounds(size, spec):
    bounds = np.array(ast.literal_eval(read(spec)))

    lower = np.zeros(size)
    upper = np.zeros(size)

    step = int(size / len(bounds))

    for i in range(len(bounds)):
        bound = bounds[i]

        lower[i * step : (i + 1) * step] = bound[0]
        upper[i * step : (i + 1) * step] = bound[1]

    return lower, upper


def parse_model(spec):
    shape = np.array(ast.literal_eval(read(spec['shape'])))
    lower, upper = parse_bounds(np.prod(shape), spec['bounds'])
    layers = parse_layers(spec['layers']) if 'layers' in spec else None
    path = spec['path'] if 'path' in spec else None

    return Model(shape, lower, upper, layers, path)


def parse_assertion(spec):
    if isinstance(spec, dict):
        # syntactic sugar definition
        return spec
    else:
        # general assertion
        input = InputStream(spec)
        lexer = AssertionLexer(input)
        token = CommonTokenStream(lexer)
        parser = AssertionParser(token)

        assertion_tree = parser.implication()
        assertion = AssertionVisitor().visitImplication(assertion_tree)

        return assertion


def parse_solver(spec):
    algorithm = spec['algorithm']

    if algorithm == 'optimize':
        solver = Optimize()
    elif algorithm == 'sprt':
        threshold = ast.literal_eval(read(spec['threshold']))
        alpha = ast.literal_eval(read(spec['alpha']))
        beta = ast.literal_eval(read(spec['beta']))
        delta = ast.literal_eval(read(spec['delta']))

        solver = SPRT(threshold, alpha, beta, delta)
    elif algorithm == 'deepcegar':
        has_ref = ast.literal_eval(read(spec['has_ref']))
        max_ref = ast.literal_eval(read(spec['max_ref']))
        ref_typ = ast.literal_eval(read(spec['ref_typ']))
        max_sus = ast.literal_eval(read(spec['max_sus']))

        solver = DeepCegar(has_ref, max_ref, ref_typ, max_sus)

    return solver


def parse_display(spec):
    mean = np.array(ast.literal_eval(read(spec['mean']))) if 'mean' in spec else np.array([0])
    std = np.array(ast.literal_eval(read(spec['std']))) if 'std' in spec else np.array([1])
    resolution = np.array(ast.literal_eval(read(spec['resolution']))) if 'resolution' in spec else np.empty(0)

    display = Display(mean, std, resolution)

    return display


def parse(spec):
    model = parse_model(spec['model'])
    assertion = parse_assertion(spec['assert'])
    solver = parse_solver(spec['solver'])
    display = parse_display(spec['display']) if 'display' in spec else None

    set_model(model)

    return model, assertion, solver, display
