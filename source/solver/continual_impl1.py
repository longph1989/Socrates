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

from nn_utils import *



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

        print(poly_out.lw)
        print(poly_out.up)

        out_lw = poly_out.lw.copy()
        out_lw[0] = poly_out.up[0]

        if np.argmax(out_lw) == 0:
            print('DeepPoly Failed!!! Use MILP!!!')
            if verify_milp(model, lst_poly, self.__write_constr_output_layer, 5):
                print('Verified!!!')
                res = True
            else:
                print('Failed!!!')
                res = False
        else:
            print('Verified!!!')
            res = True

        return res, lst_poly


    def __gen_data(self, model, lower, upper, num=3000, aux=False, is_min=True):
        xs, ys = [], []

        for i in range(num):
            x = list(generate_x(5, lower, upper))
            y = model.apply(np.array(x)).reshape(-1)

            xs.append(x)
            ys.append(y)

        if is_min:
            ys = np.argmin(ys, axis=1).reshape(-1) # get label
        else:
            ys = np.argmax(ys, axis=1).reshape(-1) # get label

        xs, ys = np.array(xs), np.array(ys)

        unique, counts = np.unique(ys, return_counts=True)
        print('Data dist = {}'.format(dict(zip(unique, counts))))

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
            print('Aux data dist = {}'.format(dict(zip(unique, counts))))

        return xs, ys


    def __train_prop(self, model, train_dataloader, prop_dataloader, loss_fn, optimizer, device, lst_poly):
        model.train()

        lamdba = 1e-3

        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.fc1.register_forward_hook(get_activation('fc1'))
        model.fc2.register_forward_hook(get_activation('fc2'))
        model.fc3.register_forward_hook(get_activation('fc3'))
        model.fc4.register_forward_hook(get_activation('fc4'))
        model.fc5.register_forward_hook(get_activation('fc5'))
        model.fc6.register_forward_hook(get_activation('fc6'))
        model.fc7.register_forward_hook(get_activation('fc7'))

        for batch, (x, y) in enumerate(prop_dataloader):
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
                loss = lamdba * (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))

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
            tensor_prop_x = torch.Tensor(prop_x.copy()) # transform to torch tensor
            tensor_prop_y = torch.Tensor(prop_y.copy()).type(torch.LongTensor)

            prop_dataset = TensorDataset(tensor_prop_x, tensor_prop_y) # create dataset
            prop_dataloader = DataLoader(prop_dataset, batch_size=100, shuffle=True) # create dataloader

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        num_of_epochs = 20
        best_acc = 0.0
            
        for epoch in range(num_of_epochs):
            print('\n------------- Epoch {} -------------\n'.format(epoch))
            if prop_x is None:
                print('Use normal loss funtion!!!')
                train(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            else:
                print('Use special loss funtion!!!')
                self.__train_prop(model, train_dataloader, prop_dataloader, nn.CrossEntropyLoss(), optimizer, device, lst_poly)
            test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)

            if best_acc < test_acc:
                best_acc = test_acc
                save_model(model, file_name)


    def __clip(self, model, fmodel, new_lst_poly, lst_poly):
        layer = 12 # the last layer

        old_weights = fmodel.layers[layer].weights
        old_bias = fmodel.layers[layer].bias

        print(old_weights.shape)
        print(old_bias.shape)

        d0, d1 = old_weights.shape[0], old_weights.shape[1]
        to_clip = [0, 2]

        min_b_lst, max_b_lst = [], []
        
        for j in range(d1):
            min_wx, max_wx = 0.0, 0.0
            for i in range(d0):
                old_weight = old_weights[i][j]
                if old_weight < 0.0:
                    max_wx += old_weight * new_lst_poly[layer].lw[i]
                    min_wx += old_weight * new_lst_poly[layer].up[i]
                elif old_weight > 0.0:
                    max_wx += old_weight * new_lst_poly[layer].up[i]
                    min_wx += old_weight * new_lst_poly[layer].lw[i]
            min_b = lst_poly[layer + 1].lw[j] - min_wx
            max_b = lst_poly[layer + 1].up[j] - max_wx
            
            min_b_lst.append(min_b)
            max_b_lst.append(max_b)

        print('min_b_lst before = {}'.format(min_b_lst))
        print('max_b_lst before = {}'.format(max_b_lst))

        min_b_lst[0] -= 10 * abs(lst_poly[layer + 1].lw[0])
        max_b_lst[2] += 10 * abs(lst_poly[layer + 1].up[2])

        print('min_b_lst after = {}'.format(min_b_lst))
        print('max_b_lst after = {}'.format(max_b_lst))

        for j in to_clip:
            if min_b_lst[j] <= max_b_lst[j]:
                fmodel.layers[layer].bias[0][j] = (min_b_lst[j] + max_b_lst[j]) / 2
                print(fmodel.layers[layer].bias[0][j])
            else:
                print('Can\'t find b!!!')
                return

        params = model.named_parameters()
        for name, param in params:
            if name == 'fc7.bias':
                for j in to_clip:
                    param.data[j] = (min_b_lst[j] + max_b_lst[j]) / 2


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

        # prop 10 bounds
        lower10 = np.array([0.269, 0.1114, -0.5, 0.2273, 0.0])
        upper10 = np.array([0.6799, 0.5, -0.4984 ,0.5, 0.5])

        len1, len2 = 5, 9
        device = 'cpu'

        ###################### Train model1 ##############################
        # for x1 in range(len1):
        #     for x2 in range(len2):
        #         print('\n=============================================\n')
        #         print('x1 = {}, x2 = {}'.format(x1, x2))

        #         model = models[x1][x2]

        #         # data from input bounds
        #         train_x0, train_y0 = self.__gen_data(model, lower0, upper0, aux=True)

        #         # data from prop 3 bounds
        #         train_x3, train_y3 = self.__gen_data(model, lower3, upper3)
        #         train_x4, train_y4 = self.__gen_data(model, lower4, upper4)

        #         # test data, use to choose best acc model
        #         test_x, test_y = self.__gen_data(model, lower0, upper0, num=10000)

        #         model0 = ACASXuNet().to(device)

        #         train_x = np.concatenate((train_x0, train_x3, train_x4), axis=0)
        #         train_y = np.concatenate((train_y0, train_y3, train_y4), axis=0)
                
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
                ###############################################################################

        ###################### Train model2 ##############################
        for x1 in range(len1):
            for x2 in range(len2):
                print('\n=============================================\n')
                print('x1 = {}, x2 = {}'.format(x1, x2))

                model = models[x1][x2]

                file_name1 = "acasxu/model1/acasxu1_200_" + str(x1) + "_" + str(x2) + ".pt"
                model1 = load_model(ACASXuNet, file_name1)

                # test data, use to choose best acc model
                test_x, test_y = self.__gen_data(model, lower0, upper0, num=10000)

                ###############################################################################
                tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
                tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

                test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
                test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

                test(model1, test_dataloader, nn.CrossEntropyLoss(), device)
                ###############################################################################

                formal_lower0, formal_upper0 = list(lower0.copy()), list(upper0.copy())
                formal_lower3, formal_upper3 = list(lower3.copy()), list(upper3.copy())
                formal_lower4, formal_upper4 = list(lower4.copy()), list(upper4.copy())

                # verify model 1 before training model 2
                formal_model1 = get_formal_model(model1, (1,5), np.array(formal_lower0), np.array(formal_upper0))
                try:
                    print('prop3 model1')
                    res, lst_poly = self.__verify(formal_model1, np.array(formal_lower3), np.array(formal_upper3))
                    print('prop4 model1')
                    self.__verify(formal_model1, np.array(formal_lower4), np.array(formal_upper4))
                except:
                    print("Error with x1 = {}, x2 = {}".format(x1, x2))
                ###############################################################################

                if not res:
                    print('Model 1 is not verified!!! Skip!!!')
                    continue

                file_name2 = "acasxu2_200_" + str(x1) + "_" + str(x2) + ".pt"

                training_mode = 'none_ext'

                if training_mode == 'none':
                    print('\nTrain with new data only!!!\n')

                    # data from prop 10 bounds, generate based on original model
                    train_x10, train_y10 = self.__gen_data(model, lower10, upper10)

                    self.__train_new_model(model1, device, train_x10, train_y10, test_x, test_y, file_name2)
                elif training_mode == 'none_ext':
                    print('\nTrain with new and old data only!!!\n')

                    # data from prop 0 bounds, generate based on original model
                    aux_train_x0, aux_train_y0 = self.__gen_data(model, lower0, upper0, num=600)
                    # data from prop 10 bounds, generate based on original model
                    train_x10, train_y10 = self.__gen_data(model, lower10, upper10)

                    train_x10 = np.concatenate((train_x10, aux_train_x0), axis=0)
                    train_y10 = np.concatenate((train_y10, aux_train_y0), axis=0)

                    self.__train_new_model(model1, device, train_x10, train_y10, test_x, test_y, file_name2)
                elif training_mode == 'data_syn':
                    print('\nTrain with data synthesis!!!\n')

                    # data from prop 3 bounds, generate based on model1
                    aux_train_x3, aux_train_y3 = self.__gen_data(formal_model1, lower3, upper3, num=5, is_min=False)
                    # data from prop 10 bounds, generate based on original model
                    train_x10, train_y10 = self.__gen_data(model, lower10, upper10)

                    train_x10 = np.concatenate((train_x10, aux_train_x3), axis=0)
                    train_y10 = np.concatenate((train_y10, aux_train_y3), axis=0)

                    self.__train_new_model(model1, device, train_x10, train_y10, test_x, test_y, file_name2)
                elif training_mode == 'continual':
                    print('\nTrain with continual certificate!!!\n')

                    # data from prop 3 bounds, generate based on model1
                    aux_train_x3, aux_train_y3 = self.__gen_data(formal_model1, lower3, upper3, num=5, is_min=False)
                    prop_x, prop_y = aux_train_x3.copy(), aux_train_y3.copy()
                    # data from prop 10 bounds, generate based on original model
                    train_x10, train_y10 = self.__gen_data(model, lower10, upper10)

                    train_x10 = np.concatenate((train_x10, aux_train_x3), axis=0)
                    train_y10 = np.concatenate((train_y10, aux_train_y3), axis=0)

                    self.__train_new_model(model1, device, train_x10, train_y10, test_x, test_y, file_name2,
                        prop_x, prop_y, lst_poly)
                elif training_mode == 'continual_ext':
                    print('\nTrain with continual certificate and extra data!!!\n')

                    # data from prop 0 bounds, generate based on original model
                    aux_train_x0, aux_train_y0 = self.__gen_data(model, lower0, upper0, num=600)

                    # data from prop 3 bounds, generate based on model1
                    aux_train_x3, aux_train_y3 = self.__gen_data(formal_model1, lower3, upper3, num=5, is_min=False)
                    prop_x, prop_y = aux_train_x3.copy(), aux_train_y3.copy()
                    # data from prop 10 bounds, generate based on original model
                    train_x10, train_y10 = self.__gen_data(model, lower10, upper10)

                    train_x10 = np.concatenate((train_x10, aux_train_x0), axis=0)
                    train_y10 = np.concatenate((train_y10, aux_train_y0), axis=0)

                    train_x10 = np.concatenate((train_x10, aux_train_x3), axis=0)
                    train_y10 = np.concatenate((train_y10, aux_train_y3), axis=0)

                    self.__train_new_model(model1, device, train_x10, train_y10, test_x, test_y, file_name2,
                        prop_x, prop_y, lst_poly)
                else:
                    assert False

                model2 = load_model(ACASXuNet, file_name2)
                print('finish model 2')

                ###############################################################################
                test(model2, test_dataloader, nn.CrossEntropyLoss(), device)
                ###############################################################################
                
                formal_model2 = get_formal_model(model2, (1,5), np.array(formal_lower0), np.array(formal_upper0))
                try:
                    print('prop3 model2')
                    res, new_lst_poly = self.__verify(formal_model2, np.array(formal_lower3), np.array(formal_upper3))

                    # self.__clip(model2, formal_model2, new_lst_poly, lst_poly)

                    # test(model2, test_dataloader, nn.CrossEntropyLoss(), device)
                    # formal_model2 = get_formal_model(model2, (1,5), np.array(formal_lower0), np.array(formal_upper0))
                    # res, new_lst_poly = self.__verify(formal_model2, np.array(formal_lower3), np.array(formal_upper3))
                except:
                    print("Error with x1 = {}, x2 = {}".format(x1, x2))
