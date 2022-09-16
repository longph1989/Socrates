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



def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name))

    return model


def print_model(model):
    for name, param in model.named_parameters():
        print(name)
        print(param.data)


def transfer_model(model1, model2, size, lbls):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    idx = lbls // 2 - 1 # idx = 1,2,3,4

    for name1, param1 in params1:
        if 'fc1' in name1:
            new_data = torch.cat((param1.data, dict_params2[name1].data[size * idx:]))
        elif 'fc2.weight' in name1 or 'fc3.weight' in name1:
            new_data0 = torch.cat((param1.data, dict_params2[name1].data[:size * idx,size * idx:]), 1)
            new_data = torch.cat((new_data0, dict_params2[name1].data[size * idx:]))
            new_data[:size * idx,size * idx:] = 0.0
        elif 'fc2.bias' in name1 or 'fc3.bias' in name1:
            new_data = torch.cat((param1.data, dict_params2[name1].data[size * idx:]))
        elif 'fc4.weight' in name1:
            new_data0 = torch.cat((param1.data, dict_params2[name1].data[:2 * idx,size * idx:]), 1)
            new_data = torch.cat((new_data0, dict_params2[name1].data[2 * idx:]))
            new_data[:2 * idx,size * idx:] = 0.0
        elif 'fc4.bias' in name1:
            new_data = torch.cat((param1.data, dict_params2[name1].data[2 * idx:]))
            
        dict_params2[name1].data.copy_(new_data)
    
    model2.load_state_dict(dict_params2)


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
        return progagate(model, idx + 1, lst_poly)


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


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def train_mnist(model, dataloader, loss_fn, optimizer, device, old_params=None, robust_dataloader=None, lst_poly_lst=None):
    size = len(dataloader.dataset)
    model.train()

    loss, loss1, loss2 = 0.0, 0.0, 0.0
    lamdba1, lambda2 = 1e-3, 1.0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss + loss_fn(pred, y)

    model.fc1.register_forward_hook(get_activation('fc1'))
    model.fc2.register_forward_hook(get_activation('fc2'))
    model.fc3.register_forward_hook(get_activation('fc3'))
    model.fc4.register_forward_hook(get_activation('fc4'))

    for i in range(4):
        for batch, (x, y) in enumerate(robust_dataloader):
            lst_poly = lst_poly_lst[batch]
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)

            k = i + 1
            layer = 'fc' + str(k)

            if i < 3:
                layer_tensor = activation[layer][:,:128]
            else:
                layer_tensor = activation[layer][:,:2]

            lower_tensor = torch.Tensor(lst_poly[2 * k - 1].lw)
            upper_tensor = torch.Tensor(lst_poly[2 * k - 1].up)
            mean_tensor = (lower_tensor + upper_tensor) / 2

            mask_lower = layer_tensor < lower_tensor
            mask_upper = layer_tensor > upper_tensor

            # square
            sum_lower = ((layer_tensor - mean_tensor) ** 2)[mask_lower].sum()
            sum_upper = ((layer_tensor - mean_tensor) ** 2)[mask_upper].sum()

            # all layers
            loss1 = loss1 + (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))

    params = model.named_parameters()
    for name, param in params:
        if 'weight' in name:
            if 'fc4' in name:
                loss2 = loss2 + ((param.data[:2,128:] - old_params[name]) ** 2).sum()
            elif 'fc1' not in name:
                loss2 = loss2 + ((param.data[:128,128:] - old_params[name]) ** 2).sum()
            else:
                loss2 = loss2 + ((param.data[:128] - old_params[name]) ** 2).sum()
        elif 'bias' in name:
            if 'fc4' in name:
                loss2 = loss2 + ((param.data[:2] - old_params[name]) ** 2).sum()
            else:
                loss2 = loss2 + ((param.data[:128] - old_params[name]) ** 2).sum()

    loss = loss + lamdba1 * loss1 + lambda2 * loss2

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    params = model.named_parameters()
    for name, param in params:
        if 'weight' in name:
            if 'fc4' in name:
                param.grad.data[:2,128:] = 0.0
            elif 'fc1' not in name:
                param.grad.data[:128,128:] = 0.0

    optimizer.step()

    # if batch % 100 == 0:
    loss, current = loss.item(), batch * len(x)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def train_robust(model, dataloader, loss_fn, optimizer, device, lst_poly_lst):
#     model.fc1.register_forward_hook(get_activation('fc1'))
#     model.fc2.register_forward_hook(get_activation('fc2'))
#     model.fc3.register_forward_hook(get_activation('fc3'))
#     model.fc4.register_forward_hook(get_activation('fc4'))

#     size = len(dataloader.dataset)
#     model.train()

#     for i in range(4):
#         # print('Train layer {}'.format(i))
#         # print_model(model)

#         for batch, (x, y) in enumerate(dataloader):
#             lst_poly = lst_poly_lst[batch]
#             x, y = x.to(device), y.to(device)

#             # Compute prediction error
#             pred = model(x)

#             k = i + 1
#             layer = 'fc' + str(k)

#             if i < 3:
#                 layer_tensor = activation[layer][:,:128]
#             else:
#                 layer_tensor = activation[layer][:,:2]

#             lower_tensor = torch.Tensor(lst_poly[2 * k - 1].lw)
#             upper_tensor = torch.Tensor(lst_poly[2 * k - 1].up)
#             mean_tensor = (lower_tensor + upper_tensor) / 2

#             mask_lower = layer_tensor < lower_tensor
#             mask_upper = layer_tensor > upper_tensor

#             # abs
#             # sum_lower = (abs(layer_tensor - lower_tensor))[mask_lower].sum()
#             # sum_upper = (abs(layer_tensor - upper_tensor))[mask_upper].sum()

#             # square
#             sum_lower = ((layer_tensor - mean_tensor) ** 2)[mask_lower].sum()
#             sum_upper = ((layer_tensor - mean_tensor) ** 2)[mask_upper].sum()

#             # all layers
#             loss = (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()

#             # modify grad here
#             params = model.named_parameters()
#             for name, param in params:
#                 if i == 0:
#                     if name == 'fc1.weight':
#                         param.grad.data[128:] = 0.0
#                     elif name == 'fc1.bias':
#                         param.grad.data[128:] = 0.0
#                     else:
#                         param.grad.data[:] = 0.0
#                 elif i == 3:
#                     if name == 'fc4.weight':
#                         param.grad.data[2:] = 0.0
#                         param.grad.data[:2,128:] = 0.0
#                     elif name == 'fc4.bias':
#                         param.grad.data[2:] = 0.0
#                     else:
#                         param.grad.data[:] = 0.0
#                 else:
#                     w_name = 'fc' + str(i + 1) + '.weight'
#                     b_name = 'fc' + str(i + 1) + '.bias'
#                     if name == w_name:
#                         param.grad.data[128:] = 0.0
#                         param.grad.data[:128,128:] = 0.0
#                     elif name == b_name:
#                         param.grad.data[128:] = 0.0
#                     else:
#                         param.grad.data[:] = 0.0

#             optimizer.step()

#             if batch % 100 == 0:
#                 loss, current = loss.item(), batch * len(x)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#         # print_model(model)


def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
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


class ACASXuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


class ContinualImpl1():
    def __write_constr_output_layer(self, prob, prev_var_idx):
        for i in range(5):
            if i != 0:
                prob.write('  x{} - x{} > 0.0\n'.format(prev_var_idx, prev_var_idx + i))

        prob.flush()


    def __verify(self, model, lower, upper):
        x0_poly = Poly()

        x0_poly.lw, x0_poly.up = lower, upper
        # just let x0_poly.le and x0_poly.ge is None
        x0_poly.shape = model.shape

        lst_poly = [x0_poly]
        poly_out = progagate(model, 0, lst_poly)

        out_lw = poly_out.lw.copy()
        out_lw[0] = poly_out.up[0]

        if np.argmax(out_lw) == 0:
            print('DeepPoly Failed!!! Use MILP!!!')
            if verify_milp(model, lst_poly, self.__write_constr_output_layer, 5):
                print('Verified!!!')
            else:
                print('Failed!!!')
        else:
            print('Verified!!!')

        return lst_poly


    def __gen_train_data(self, model, lower, upper, num=3000, aux=False, is_min=True):
        xs, ys = [], []

        for i in range(num):
            x = list(generate_x(5, lower, upper))
            y = model.apply(np.array(x)).reshape(-1)

            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        if is_min:
            ys = np.argmin(ys, axis=1).reshape(-1) # get label
        else:
            ys = np.argmax(ys, axis=1).reshape(-1) # get label

        unique, counts = np.unique(ys, return_counts=True)
        print('Train dist = {}'.format(dict(zip(unique, counts))))

        if aux:
            aux_x, aux_y = [], []

            for i in range(len(xs)):
                if ys[i] != 0:
                    for j in range(9):
                        aux_x.append(xs[i])
                        aux_y.append(ys[i])

            xs = np.concatenate((xs, np.array(aux_x)), axis=0)
            ys = np.concatenate((ys, np.array(aux_y)), axis=0)

            unique, counts = np.unique(ys, return_counts=True)
            print('Aux train dist = {}'.format(dict(zip(unique, counts))))

        return xs, ys


    def __gen_test_data(self, model, lower, upper, num=100000):
        xs, ys = [], []

        for i in range(num):
            x = list(generate_x(5, lower, upper))
            y = model.apply(np.array(x)).reshape(-1)

            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.argmin(ys, axis=1).reshape(-1) # get label

        unique, counts = np.unique(ys, return_counts=True)
        print('Test dist = {}'.format(dict(zip(unique, counts))))

        return xs, ys


    def __train_prop(self, model, train_dataloader, prop_dataloader, loss_fn, optimizer, device, lst_poly):
        model.train()

        loss, loss1 = 0.0, 0.0
        lamdba1 = 1e-3

        for batch1, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss + loss_fn(pred, y)

        for batch2, (x, y) in enumerate(prop_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss + loss_fn(pred, y)

        model.fc1.register_forward_hook(get_activation('fc1'))
        model.fc2.register_forward_hook(get_activation('fc2'))
        model.fc3.register_forward_hook(get_activation('fc3'))
        model.fc4.register_forward_hook(get_activation('fc4'))
        model.fc5.register_forward_hook(get_activation('fc5'))
        model.fc6.register_forward_hook(get_activation('fc6'))
        model.fc7.register_forward_hook(get_activation('fc7'))

        for batch3, (x, y) in enumerate(prop_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)

            for i in range(7):
                layer = 'fc' + str(i + 1)
                layer_tensor = activation[layer]

                lower_tensor = torch.Tensor(lst_poly[2 * i + 1].lw)
                upper_tensor = torch.Tensor(lst_poly[2 * i + 1].up)
                mean_tensor = (lower_tensor + upper_tensor) / 2

                mask_lower = layer_tensor < lower_tensor
                mask_upper = layer_tensor > upper_tensor

                # square
                sum_lower = ((layer_tensor - mean_tensor) ** 2)[mask_lower].sum()
                sum_upper = ((layer_tensor - mean_tensor) ** 2)[mask_upper].sum()

                # all layers
                loss1 = loss1 + (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))

        loss = loss / (batch1 + batch2 + 2)
        loss1 = loss1 / (batch3 + 1)

        loss = loss + lamdba1 * loss1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def __train_new_model(self, model, device, train_x, train_y, test_x, test_y, file_name, prop_x=None, prop_y=None, lst_poly=None):
        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        if prop_x is not None:
            print('\nTrain with continual certificate!!!\n')

            tensor_prop_x = torch.Tensor(prop_x.copy()) # transform to torch tensor
            tensor_prop_y = torch.Tensor(prop_y.copy()).type(torch.LongTensor)

            prop_dataset = TensorDataset(tensor_prop_x, tensor_prop_y) # create dataset
            prop_dataloader = DataLoader(prop_dataset, batch_size=100, shuffle=True) # create dataloader

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        num_of_epochs = 200
        best_acc = 0.0
            
        for epoch in range(num_of_epochs):
            print('\n------------- Epoch {} -------------\n'.format(epoch))
            if prop_x is not None:
                # use special loss function
                self.__train_prop(model, train_dataloader, prop_dataloader, nn.CrossEntropyLoss(), optimizer, device, lst_poly)
            else:
                train(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)

            if best_acc < test_acc:
                best_acc = test_acc
                save_model(model, file_name)


    def solve(self, models, assertion, display=None):
        # input bounds
        lower0 = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper0 = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        # prop 3 bounds
        lower3 = np.array([-0.3035, -0.0095, 0.4934, 0.3, 0.3])
        upper3 = np.array([-0.2986, 0.0095, 0.5, 0.5, 0.5])

        # prop 4 bounds
        lower4 = np.array([-0.3035, -0.0095, 0.0, 0.3182, 0.0833])
        upper4 = np.array([-0.2986, 0.0095, 0.0, 0.5, 0.1667])

        len1, len2 = 5, 9
        device = 'cpu'

        ###################### Train model1 ##############################
        # for x1 in range(len1):
        #     for x2 in range(len2):
        #         print('\n=============================================\n')
        #         print('x1 = {}, x2 = {}'.format(x1, x2))

        #         model = models[x1][x2]

        #         # data from input bounds
        #         train_x0, train_y0 = self.__gen_train_data(model, lower0, upper0, aux=True)

        #         # data from prop 3 bounds
        #         train_x3, train_y3 = self.__gen_train_data(model, lower3, upper3)

        #         # test data, use to choose best acc model
        #         test_x, test_y = self.__gen_test_data(model, lower0, upper0)

        #         model0 = ACASXuNet().to(device)

        #         train_x = np.concatenate((train_x0, train_x3), axis=0)
        #         train_y = np.concatenate((train_y0, train_y3), axis=0)
                
        #         file_name1 = "acasxu1_200_" + str(x1) + "_" + str(x2) + ".pt"
        #         self.__train_new_model(model0, device, train_x, train_y, test_x, test_y, file_name1)

        #         model1 = load_model(ACASXuNet, file_name1)
        #         print('finish model 1')

        #         ###############################################################################
        #         tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        #         tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        #         test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        #         test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        #         test(model1, test_dataloader, nn.CrossEntropyLoss(), device)
        #         ###############################################################################

        ###################### Verify model1 ##############################
        # for x1 in range(5):
        #     for x2 in range(9):
        #         print('\n=============================================\n')
        #         print('x1 = {}, x2 = {}'.format(x1, x2))

        #         file_name1 = "acasxu1_200_" + str(x1) + "_" + str(x2) + ".pt"

        #         model1 = load_model(ACASXuNet, file_name1)
        #         print('finish model 1')

        #         formal_lower0, formal_upper0 = list(lower0.copy()), list(upper0.copy())
        #         formal_lower3, formal_upper3 = list(lower3.copy()), list(upper3.copy())

        #         formal_model1 = get_formal_model(model1, (1,5), np.array(formal_lower0), np.array(formal_upper0))
        #         try:
        #             lst_poly = self.__verify(formal_model1, np.array(formal_lower3), np.array(formal_upper3))
        #         except:
        #             print("Error with x1 = {}, x2 = {}".format(x1, x2))

        ###################### Train model2 ##############################
        for x1 in range(len1):
            for x2 in range(len2):
                print('\n=============================================\n')
                print('x1 = {}, x2 = {}'.format(x1, x2))

                model = models[x1][x2]

                file_name1 = "acasxu/model1/acasxu1_200_" + str(x1) + "_" + str(x2) + ".pt"
                model1 = load_model(ACASXuNet, file_name1)

                formal_lower0, formal_upper0 = list(lower0.copy()), list(upper0.copy())
                formal_lower3, formal_upper3 = list(lower3.copy()), list(upper3.copy())
                formal_lower4, formal_upper4 = list(lower4.copy()), list(upper4.copy())

                formal_model1 = get_formal_model(model1, (1,5), np.array(formal_lower0), np.array(formal_upper0))
                try:
                    print('prop3 model1')
                    lst_poly = self.__verify(formal_model1, np.array(formal_lower3), np.array(formal_upper3))
                    print('prop4 model1')
                    self.__verify(formal_model1, np.array(formal_lower4), np.array(formal_upper4))
                except:
                    print("Error with x1 = {}, x2 = {}".format(x1, x2))

                # data from prop 3 bounds, generate based on model1
                aux_train_x3, aux_train_y3 = self.__gen_train_data(formal_model1, lower3, upper3, num=300, is_min=False)

                # data from prop 4 bounds, generate based on original model
                train_x4, train_y4 = self.__gen_train_data(model, lower4, upper4)

                # test data, use to choose best acc model
                test_x, test_y = self.__gen_test_data(model, lower0, upper0)

                ###############################################################################
                tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
                tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

                test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
                test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

                test(model1, test_dataloader, nn.CrossEntropyLoss(), device)
                ###############################################################################
                
                file_name2 = "acasxu2_200_" + str(x1) + "_" + str(x2) + ".pt"
                self.__train_new_model(model1, device, train_x4, train_y4, test_x, test_y, file_name2,
                        aux_train_x3, aux_train_y3, lst_poly)

                model2 = load_model(ACASXuNet, file_name2)
                print('finish model 2')

                ###############################################################################
                test(model2, test_dataloader, nn.CrossEntropyLoss(), device)
                ###############################################################################
                
                formal_model2 = get_formal_model(model2, (1,5), np.array(formal_lower0), np.array(formal_upper0))
                try:
                    print('prop3 model2')
                    self.__verify(formal_model2, np.array(formal_lower3), np.array(formal_upper3))
                    print('prop4 model2')
                    self.__verify(formal_model2, np.array(formal_lower4), np.array(formal_upper4))
                except:
                    print("Error with x1 = {}, x2 = {}".format(x1, x2))


class MNISTNet(nn.Module):
    def __init__(self, lbl):
        super().__init__()
        self.fc1 = nn.Linear(784, 128 * lbl // 2)
        self.fc2 = nn.Linear(128 * lbl // 2, 128 * lbl // 2)
        self.fc3 = nn.Linear(128 * lbl // 2, 128 * lbl // 2)
        self.fc4 = nn.Linear(128 * lbl // 2, lbl)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


class ContinualImpl2():
    def __init__(self):
        self.device = torch.device("cpu")

        self.train_kwargs = {'batch_size': 100}
        self.test1_kwargs = {'batch_size': 1}
        self.test1000_kwargs = {'batch_size': 1000}

        self.transform = transforms.ToTensor()
        self.eps = 0.01


    def __mask_off(self, train_index):
        for index in range(len(train_index)):
            if train_index[index]:
                if random.random() > 0.2:
                    train_index[index] = False


    def __gen_more_data(self, robust_lst, target_lst):
        aux_robust_lst, aux_target_lst = [], []

        for i in range(len(target_lst)):
            img = robust_lst[i]
            lower_i = (img - self.eps).clip(0, 1).reshape(-1)
            upper_i = (img + self.eps).clip(0, 1).reshape(-1)

            for j in range(20):
                aux_img = generate_x(784, lower_i, upper_i)
                aux_robust_lst.append(aux_img)
                # aux_robust_lst.append(img.copy())
                aux_target_lst.append(target_lst[i])

        return aux_robust_lst, aux_target_lst


    def __train_iteration(self):
        masked_index_lst = []
        robust_lst, target_lst, lst_poly_lst = [], [], []

        for lbl in range(2, 5, 2):
            print(lbl)

            train_dataset = datasets.MNIST('../data', train=True, download=True, transform=self.transform)
            test_dataset = datasets.MNIST('../data', train=False, transform=self.transform)

            for i in range(lbl - 2, lbl):
                if i % 2 == 0:
                    train_index, mask_index = train_dataset.targets == i, train_dataset.targets == i
                else:
                    train_index = train_index | (train_dataset.targets == i)
                    mask_index = mask_index | (train_dataset.targets == i)

                if i == 0:
                    test_index = test_dataset.targets == 0
                else:
                    test_index = test_index | (test_dataset.targets == i)

            for masked_index in masked_index_lst:
                train_index = train_index | masked_index

            train_dataset.data, train_dataset.targets = train_dataset.data[train_index], train_dataset.targets[train_index]
            test_dataset.data, test_dataset.targets = test_dataset.data[test_index], test_dataset.targets[test_index]

            train_dataloader = torch.utils.data.DataLoader(train_dataset, **self.train_kwargs)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test1000_kwargs)

            if lbl > 2: old_model = model
            model = MNISTNet(lbl).to(self.device)
            if lbl > 2: transfer_model(old_model, model)

            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            num_of_epochs = 10 * lbl

            if lbl == 2:
                model = load_model(MNISTNet, 'mnist1.pt', lbl)
                # best_acc = 0.0

                # for epoch in range(num_of_epochs):
                #     print('\n------------- Epoch {} -------------\n'.format(epoch))
                #     train(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, self.device)
                #     test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), self.device)

                #     if test_acc > best_acc:
                #         best_acc = test_acc
                #         save_model(model, 'mnist1.pt')
            else:
                # model = load_model(MNISTNet, 'mnist2_robust.pt', lbl)
                if len(robust_lst) > 0:
                    aux_robust_lst, aux_target_lst = self.__gen_more_data(robust_lst, target_lst)
                    print('more train with len = {}'.format(len(aux_robust_lst)))

                    robust_train_x = torch.Tensor(np.array(aux_robust_lst)) # transform to torch tensor
                    robust_train_y = torch.Tensor(np.array(aux_target_lst)).type(torch.LongTensor)

                    robust_dataset = TensorDataset(robust_train_x, robust_train_y) # create dataset
                    robust_dataloader = DataLoader(robust_dataset, batch_size=20, shuffle=False) # create dataloader

                best_acc = 0.0
                old_params = {}
                for name, param in old_model.named_parameters():
                    old_params[name] = param.data.clone()

                for epoch in range(num_of_epochs):
                    print('\n------------- Epoch {} -------------\n'.format(epoch))
                    train_mnist(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, self.device, old_params, robust_dataloader, lst_poly_lst)
                    test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), self.device)

                    if test_acc > best_acc:
                        best_acc = test_acc
                        if lbl == 4:
                            save_model(model, 'mnist2.pt')
                        elif lbl == 6:
                            save_model(model, 'mnist3.pt')
                        elif lbl == 8:
                            save_model(model, 'mnist4.pt')
                        elif lbl == 10:
                            save_model(model, 'mnist5.pt')
                
                    # if len(robust_lst) > 0:
                    #     # train_mnist(model, robust_dataloader, nn.CrossEntropyLoss(), optimizer, self.device)
                    #     train_robust(model, robust_dataloader, nn.CrossEntropyLoss(), optimizer, self.device, lst_poly_lst)
                
            self.__verify_iteration(robust_lst, target_lst, lst_poly_lst, lbl)
            self.__mask_off(mask_index)
            masked_index_lst.append(mask_index)


    def __write_constr_output_layer(self, y0, y, prob, prev_var_idx):
        prob.write('  x{} - x{} > 0.0\n'.format(prev_var_idx + y, prev_var_idx + y0))
        prob.flush()


    def __verify(self, model, lower, upper, y0):
        x0_poly = Poly()

        x0_poly.lw, x0_poly.up = lower, upper
        # just let x0_poly.le and x0_poly.ge is None
        x0_poly.shape = model.shape

        lst_poly = [x0_poly]
        poly_out = progagate(model, 0, lst_poly)
        no_neurons = len(poly_out.lw)

        for y in range(no_neurons):
            if y != y0 and poly_out.lw[y0] <= poly_out.up[y]:
                poly_res = Poly()

                poly_res.lw = np.zeros(1)
                poly_res.up = np.zeros(1)

                poly_res.le = np.zeros([1, no_neurons + 1])
                poly_res.ge = np.zeros([1, no_neurons + 1])

                poly_res.ge[0,y0] = 1
                poly_res.ge[0,y] = -1

                poly_res.back_substitute(lst_poly)

                if poly_res.lw[0] <= 0:
                    partial_output = partial(self.__write_constr_output_layer, y0, y)
                    if not verify_milp(model, lst_poly, partial_output, 784):
                        return False, lst_poly

        return True, lst_poly


    def __verify_iteration(self, robust_lst, target_lst, lst_poly_lst, lbl):
        print(lbl)

        test_dataset = datasets.MNIST('../data', train=False, transform=self.transform)

        for i in range(0, lbl):
            if i == 0:
                test_index = test_dataset.targets == 0
            else:
                test_index = test_index | (test_dataset.targets == i)

        test_dataset.data, test_dataset.targets = test_dataset.data[test_index], test_dataset.targets[test_index]
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test1000_kwargs)

        if lbl == 2:
            model = load_model(MNISTNet, 'mnist1.pt', lbl)
        elif lbl == 4:
            model = load_model(MNISTNet, 'mnist2.pt', lbl)
        elif lbl == 6:
            model = load_model(MNISTNet, 'mnist3.pt', lbl)
        elif lbl == 8:
            model = load_model(MNISTNet, 'mnist4.pt', lbl)
        elif lbl == 10:
            model = load_model(MNISTNet, 'mnist5.pt', lbl)

        shape, lower, upper = (1,784), np.zeros(784), np.ones(784)
        formal_model = get_formal_model(model, shape, lower, upper)

        test(model, test_dataloader, nn.CrossEntropyLoss(), self.device)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test1_kwargs)

        print('robust len = {}'.format(len(robust_lst)))
        pass_cnt, fail_cnt = 0, 0

        for i in range(len(robust_lst)):
            img, target = robust_lst[i], target_lst[i]

            lower_i, upper_i = (img - self.eps).reshape(-1), (img + self.eps).reshape(-1)
            lower_i = np.maximum(lower_i, formal_model.lower)
            upper_i = np.minimum(upper_i, formal_model.upper)

            if self.__verify(formal_model, lower_i, upper_i, target):
                pass_cnt += 1
            else:
                fail_cnt += 1

        print('pass_cnt = {}, fail_cnt = {}, percent = {}'.format(pass_cnt, fail_cnt,
            pass_cnt / len(robust_lst) if len(robust_lst) > 0 else 0))

        for data, target in test_dataloader:
            img = data.numpy().reshape(1, 784)

            lower_i, upper_i = (img - self.eps).reshape(-1), (img + self.eps).reshape(-1)
            lower_i = np.maximum(lower_i, formal_model.lower)
            upper_i = np.minimum(upper_i, formal_model.upper)
            target = target.numpy()[0]

            if target >= lbl - 2:
                res, lst_poly = self.__verify(formal_model, lower_i, upper_i, target)
                if res:
                    robust_lst.append(img)
                    target_lst.append(target)
                    lst_poly_lst.append(lst_poly)

                    if len(robust_lst) == lbl * 10:
                        print('enough')
                        break


    def solve(self, models, assertion, display=None):
        self.__train_iteration()
        # self.__verify_iteration()
