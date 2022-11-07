import torch
import random
import numpy as np

from model.lib_models import *
from model.lib_layers import *

from poly_utils import *
from solver.refinement_impl import Poly

from utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import gurobipy as gp
from gurobipy import GRB

import math
import ast



def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name))

    return model


def print_model(model, names=None):
    for name, param in model.named_parameters():
        if names is None or name in names:
            print(name)
            print(param.data)


def get_layers(model):
    layers, params = list(), list(model.named_parameters())

    for i in range(len(params)):
        name, param = params[i]
        if 'weight' in name:
            weight = np.array(param.data)
        elif 'bias' in name:
            bias = np.array(param.data)

            layers.append(Linear(weight, bias, None))
            if i < len(params) - 1: # last layer
                layers.append(Function('relu', None))

    return layers


def get_formal_model(model, shape, lower, upper):
    lower, upper = lower.copy(), upper.copy()
    layers = get_layers(model)
    
    return Model(shape, lower, upper, layers, None)


def progagate(model, idx, lst_poly):
    if idx == len(model.layers):
        poly_out = lst_poly[idx]
        return poly_out
    else:
        poly_next = model.forward(lst_poly[idx], idx, lst_poly)

        lw_next = poly_next.lw
        up_next = poly_next.up

        if np.any(up_next < lw_next):
            raise Exception("Unreachable states!!!") # unreachable states

        lst_poly.append(poly_next)
        poly_out = progagate(model, idx + 1, lst_poly)

        return poly_out


def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, dataloader, loss_fn, device, debug=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print(size)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def write_constr_hidden_layers(prob, coefs, const, prev_var_idx, curr_var_idx):
    prob.write('  x{}'.format(curr_var_idx))

    for i in range(len(coefs)):
        coef = coefs[i]
        var_idx = prev_var_idx + i

        if coef > 0:
            prob.write(' + {} x{}'.format(coef, var_idx))
        elif coef < 0:
            prob.write(' - {} x{}'.format(abs(coef), var_idx))

    prob.write(' = {}\n'.format(const))
    prob.flush()


def write_constr_relu_layers(prob, prev_var_idx, curr_var_idx, poly, number_of_neurons):
    for i in range(number_of_neurons):
        pvar_idx, cvar_idx = prev_var_idx + i, curr_var_idx + i

        ge_coefs = -poly.ge[i][i]
        ge_const = poly.ge[i][-1]

        le_coefs = -poly.le[i][i]
        le_const = poly.le[i][-1]

        prob.write('  x{}'.format(cvar_idx))
        if ge_coefs > 0.0:
            prob.write(' + {} x{}'.format(ge_coefs, pvar_idx))
        elif ge_coefs < 0.0:
            prob.write(' - {} x{}'.format(abs(ge_coefs), pvar_idx))
        prob.write(' >= {}\n'.format(ge_const))

        prob.write('  x{}'.format(cvar_idx))
        if le_coefs > 0.0:
            prob.write(' + {} x{}'.format(le_coefs, pvar_idx))
        elif le_coefs < 0.0:
            prob.write(' - {} x{}'.format(abs(le_coefs), pvar_idx))
        prob.write(' <= {}\n'.format(le_const))
    prob.flush()


def write_constr(prob, model, lst_poly, input_len):
    prev_var_idx, curr_var_idx = 0, input_len

    for i in range(len(model.layers)):
        layer = model.layers[i]

        if layer.is_linear_layer():
            weights = layer.weights.transpose(1, 0)
            bias = layer.bias.reshape(-1)

            for i in range(len(bias)):
                coefs, const = -weights[i], bias[i]
                write_constr_hidden_layers(prob, coefs, const, prev_var_idx, curr_var_idx + i)

            prev_var_idx = curr_var_idx
            curr_var_idx = curr_var_idx + len(bias)
        else:
            # the old bias value
            write_constr_relu_layers(prob, prev_var_idx, curr_var_idx, lst_poly[i+1], len(bias))

            prev_var_idx = curr_var_idx
            curr_var_idx = curr_var_idx + len(bias)

    return prev_var_idx


def write_bounds(prob, lst_poly):
    i = 0

    for poly in lst_poly:
        for j in range(len(poly.lw)):
            prob.write('  {} <= x{} <= {}\n'.format(poly.lw[j], i, poly.up[j]))
            i = i + 1

    prob.flush()


def write_problem(model, lst_poly, output_constr, input_len):
    filename = 'prob.lp'
    prob = open(filename, 'w')

    prob.write('Minimize\n')
    prob.write('  0\n')

    prob.write('Subject To\n')
    prev_var_idx = write_constr(prob, model, lst_poly, input_len)
    output_constr(prob, prev_var_idx)

    prob.write('Bounds\n')
    write_bounds(prob, lst_poly)

    prob.write('End\n')

    prob.flush()
    prob.close()


def verify_milp(model, lst_poly, output_constr, input_len):
    write_problem(model, lst_poly, output_constr, input_len)

    filename = 'prob.lp'
    opt = gp.read(filename)
    opt.setParam(GRB.Param.DualReductions, 0)

    opt.optimize()

    if opt.status == GRB.INFEASIBLE:
        print('Infeasible')
        return True
    elif opt.status == GRB.OPTIMAL:
        print('Satisfiable')
        return False
    else:
        print('Unknown')
        return False
