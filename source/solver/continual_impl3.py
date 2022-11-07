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

import time



class CensusNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
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
        output = x # cross entropy in pytorch already includes softmax
        return output


class ContinualImpl3():
    def __get_data(self, input_file, label_file):
        input_file = open(input_file, 'r')
        input_data = []
        for line in input_file.readlines():
            input_data.append(ast.literal_eval(line))
        input_data = np.array(input_data)

        label_file = open(label_file, 'r')
        label_data = np.array(ast.literal_eval(label_file.readline()))

        return input_data, label_data

    def __gen_data(self, fair_lst, target_lst, num_of_features, gender_idx):
        xs, ys = [], []

        for i in range(len(target_lst)):
            data = fair_lst[i]
            lower_i, upper_i = data.copy().reshape(-1), data.copy().reshape(-1)
            lower_i[gender_idx], upper_i[gender_idx] = 0.0, 1.0

            for j in range(10):
                x = generate_x(num_of_features, lower_i, upper_i)

                xs.append(x)
                ys.append(target_lst[i])

        unique, counts = np.unique(ys, return_counts=True)
        print('Data dist = {}'.format(dict(zip(unique, counts))))

        return np.array(xs), np.array(ys)


    def __train_fair(self, model, train_dataloader, fair_dataloader, loss_fn, optimizer, device, lst_poly_lst):
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

        for batch, (x, y) in enumerate(fair_dataloader):
            x, y = x.to(device), y.to(device)
            lst_poly = lst_poly_lst[batch]

            # Compute prediction error
            pred = model(x)
            loss = 0.0

            for i in range(6):
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
                loss = loss + lamdba * (sum_lower + sum_upper) / (len(layer_tensor) * len(layer_tensor[0]))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def __write_constr_output_layer(self, y0, y, prob, prev_var_idx):
        prob.write('  x{} - x{} > 0.0\n'.format(prev_var_idx + y, prev_var_idx + y0))
        prob.flush()


    def __verify(self, model, input_size, lower, upper, y0):
        x0_poly = Poly()
        x0_poly.lw, x0_poly.up = lower, upper
        # just let x0_poly.le and x0_poly.ge is None
        x0_poly.shape = model.shape

        lst_poly = [x0_poly]
        poly_out = progagate(model, 0, lst_poly)

        # print('lw = {}'.format(poly_out.lw))
        # print('up = {}'.format(poly_out.up))

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
                    if not verify_milp(model, lst_poly, partial_output, input_size):
                        return False, lst_poly

        return True, lst_poly


    def __verify_iteration(self, model, num_of_features, gender_idx, fair_lst=None, target_lst=None):
        shape, lower, upper = (1,13), np.zeros(13), np.ones(13)
        formal_model = get_formal_model(model, shape, lower, upper)

        if fair_lst is None:
            test_x, test_y = self.__get_data('test_data/input.txt', 'test_data/label.txt')

            tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
            tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

            test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
            test_dataloader = DataLoader(test_dataset, batch_size=1) # create dataloader

            fair_lst, target_lst, lst_poly_lst = [], [], []

            for data, target in test_dataloader:
                data = data.numpy().reshape(1, num_of_features)
                lower_i, upper_i = data.copy().reshape(-1), data.copy().reshape(-1)
                lower_i[gender_idx], upper_i[gender_idx] = 0.0, 1.0

                target = target.numpy()[0]

                res, lst_poly = self.__verify(formal_model, num_of_features, lower_i, upper_i, target)
                if res:
                    fair_lst.append(data)
                    target_lst.append(target)
                    lst_poly_lst.append(lst_poly)

                    if len(fair_lst) == 10:
                        print('Enough fair samples')
                        print(target_lst)
                        break

            return fair_lst, target_lst, lst_poly_lst
        else:
            pass_cnt, fail_cnt = 0, 0

            for idx, data in enumerate(fair_lst):
                lower_i, upper_i = data.copy().reshape(-1), data.copy().reshape(-1)
                lower_i[gender_idx], upper_i[gender_idx] = 0.0, 1.0

                target = target_lst[idx]

                res, lst_poly = self.__verify(formal_model, num_of_features, lower_i, upper_i, target)

                if res: pass_cnt += 1
                else: fail_cnt += 1

            print('pass_cnt = {}, fail_cnt = {}, percent = {}'.format(pass_cnt, fail_cnt,
                pass_cnt / len(fair_lst) if len(fair_lst) > 0 else 0))


    def __train_model(self, model, train_x, train_y, test_x, test_y, device, file_name, fair_x=None, fair_y=None, lst_poly_lst=None):
        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        num_of_epochs = 20

        if fair_x is not None:
            tensor_fair_x = torch.Tensor(fair_x.copy()) # transform to torch tensor
            tensor_fair_y = torch.Tensor(fair_y.copy()).type(torch.LongTensor)

            fair_dataset = TensorDataset(tensor_fair_x, tensor_fair_y) # create dataset
            fair_dataloader = DataLoader(fair_dataset, batch_size=10, shuffle=False) # create dataloader

        best_acc = 0.0

        start = time.time()
        for epoch in range(num_of_epochs):
            print('\n------------- Epoch {} -------------\n'.format(epoch))
            if fair_x is None:
                print('Use normal loss funtion!!!')
                train(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            else:
                print('Use special loss funtion!!!')
                self.__train_fair(model, train_dataloader, fair_dataloader, nn.CrossEntropyLoss(), optimizer, device, lst_poly_lst)
            test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)

            if best_acc < test_acc:
                best_acc = test_acc
                save_model(model, file_name)
        end = time.time()

        return end - start


    def __is_discriminative(self, model, x):
        pred_x = model(x)

        xp = x.detach().clone()
        xp[0][8] = 1 - xp[0][8]

        pred_xp = model(xp)

        y, yp = torch.argmax(pred_x).item(), torch.argmax(pred_xp).item()
        
        if y != yp:
            return True, xp, y
        else:
            return False, None, None


    def __adf(self, model, train_x, train_y):
        # sample_idx = random.sample(range(len(train_x)), 5000)
        # lamb = 0.01

        # adf_x, adf_y = [], []

        # for i in sample_idx:
        #     x, y = train_x[i], train_y[i]

        #     x = torch.Tensor(x.reshape(1, 13))
        #     y = torch.Tensor(np.array([y]))
        #     y = y.type(torch.LongTensor)

        #     for _ in range(10):
        #         x.requires_grad = True
        #         pred_x = model(x)
        #         loss = F.cross_entropy(pred_x, y)
        #         loss.backward()
        #         x = x + lamb * x.grad.data.sign()

        #         res, xp, yp = self.__is_discriminative(model, x)
        #         if res:
        #             adf_x.append(x.detach().clone().numpy().reshape(-1))
        #             adf_x.append(xp.detach().clone().numpy().reshape(-1))
        #             adf_y.append(yp)
        #             adf_y.append(yp)
        #             break

        #         x = x.detach().clone()

        adf_x, adf_y = self.__get_data('adf_data/input.txt', 'adf_data/label.txt')

        print('adf samples = {}'.format(len(adf_x)))

        return np.array(adf_x), np.array(adf_y)


    def solve(self, models, assertion, display=None):
        training_mode = 'continual_ext'

        device = 'cpu'
        train_kwargs = {'batch_size': 100}
        test_kwargs = {'batch_size': 1000}

        train_x, train_y = self.__get_data('train_data/input.txt', 'train_data/label.txt')
        test_x, test_y = self.__get_data('test_data/input.txt', 'test_data/label.txt')

        print('training samples = {}'.format(len(train_x)))

        num_of_features = 13
        gender_idx = 8

        model = CensusNet(num_of_features).to(device)
        # self.__train_model(model, train_x, train_y, test_x, test_y, device, 'census1.pt')

        model = load_model(CensusNet, 'census/census1.pt', num_of_features)

        ##################################################################
        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        test(model, test_dataloader, nn.CrossEntropyLoss(), device)
        ##################################################################

        fair_lst, target_lst, lst_poly_lst = self.__verify_iteration(model, num_of_features, gender_idx)
        total_time = 0.0

        if training_mode == 'none':
            print('\nTrain with new data only!!!\n')

            new_x, new_y = self.__adf(model, train_x, train_y)
            total_time += self.__train_model(model, new_x, new_y, test_x, test_y, device, 'census2.pt')
        elif training_mode == 'none_ext':
            print('\nTrain with new and old data only!!!\n')

            old_x, old_y = self.__get_data('train_data/input.txt', 'train_data/label.txt')
            print(len(old_x))

            sample_idx = random.sample(range(len(train_x)), 6500)
            old_x, old_y = old_x[sample_idx], old_y[sample_idx]

            new_x, new_y = self.__adf(model, train_x, train_y)

            new_x = np.concatenate((new_x, old_x), axis=0)
            new_y = np.concatenate((new_y, old_y), axis=0)

            total_time += self.__train_model(model, new_x, new_y, test_x, test_y, device, 'census2.pt')
        elif training_mode == 'data_syn':
            print('\nTrain with data synthesis!!!\n')

            new_x, new_y = self.__adf(model, train_x, train_y)
            aux_x, aux_y = self.__gen_data(fair_lst, target_lst, num_of_features, gender_idx)

            print('aux samples = {}'.format(len(aux_x)))

            new_x = np.concatenate((new_x, aux_x), axis=0)
            new_y = np.concatenate((new_y, aux_y), axis=0)

            total_time += self.__train_model(model, new_x, new_y, test_x, test_y, device, 'census2.pt')
        elif training_mode == 'continual':
            print('\nTrain with continual certificate!!!\n')

            new_x, new_y = self.__adf(model, train_x, train_y)
            aux_x, aux_y = self.__gen_data(fair_lst, target_lst, num_of_features, gender_idx)

            print('aux samples = {}'.format(len(aux_x)))

            new_x = np.concatenate((new_x, aux_x.copy()), axis=0)
            new_y = np.concatenate((new_y, aux_y.copy()), axis=0)

            total_time += self.__train_model(model, new_x, new_y, test_x, test_y, device, 'census2.pt', aux_x, aux_y, lst_poly_lst)
        elif training_mode == 'continual_ext':
            print('\nTrain with continual certificate and extra data!!!\n')

            old_x, old_y = self.__get_data('train_data/input.txt', 'train_data/label.txt')
            print(len(old_x))

            sample_idx = random.sample(range(len(train_x)), 6500)
            old_x, old_y = old_x[sample_idx], old_y[sample_idx]

            new_x, new_y = self.__adf(model, train_x, train_y)
            aux_x, aux_y = self.__gen_data(fair_lst, target_lst, num_of_features, gender_idx)

            print('aux samples = {}'.format(len(aux_x)))

            new_x = np.concatenate((new_x, old_x), axis=0)
            new_y = np.concatenate((new_y, old_y), axis=0)

            new_x = np.concatenate((new_x, aux_x.copy()), axis=0)
            new_y = np.concatenate((new_y, aux_y.copy()), axis=0)

            total_time += self.__train_model(model, new_x, new_y, test_x, test_y, device, 'census2.pt', aux_x, aux_y, lst_poly_lst)
        else:
            assert False

        total_time = round(total_time)
        minute = total_time // 60
        second = total_time - 60 * minute

        print('Time = {}m{}s'.format(minute, second))

        model = load_model(CensusNet, 'census2.pt', num_of_features)

        ##################################################################
        test(model, test_dataloader, nn.CrossEntropyLoss(), device)
        ##################################################################

        self.__verify_iteration(model, num_of_features, gender_idx, fair_lst, target_lst)
