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



class MNISTNet(nn.Module):
    def __init__(self, size, num_of_lbls):
        super().__init__()
        self.fc1 = nn.Linear(784, size * num_of_lbls // 2)
        self.fc2 = nn.Linear(size * num_of_lbls // 2, size * num_of_lbls // 2)
        self.fc3 = nn.Linear(size * num_of_lbls // 2, size * num_of_lbls // 2)
        self.fc4 = nn.Linear(size * num_of_lbls // 2, num_of_lbls)

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


class CIFAR10Net(nn.Module):
    def __init__(self, size, num_of_lbls):
        super().__init__()
        self.fc1 = nn.Linear(3072, size * num_of_lbls // 2)
        self.fc2 = nn.Linear(size * num_of_lbls // 2, size * num_of_lbls // 2)
        self.fc3 = nn.Linear(size * num_of_lbls // 2, size * num_of_lbls // 2)
        self.fc4 = nn.Linear(size * num_of_lbls // 2, num_of_lbls)

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
    def __gen_data(self, dataset, robust_lst, target_lst, eps, num_samples):
        xs, ys = [], []

        for i in range(len(target_lst)):
            img = robust_lst[i]
            lower_i = (img - eps).clip(0, 1).reshape(-1)
            upper_i = (img + eps).clip(0, 1).reshape(-1)

            for j in range(num_samples):
                if dataset == 'mnist':
                    aux_img = generate_x(784, lower_i, upper_i)
                elif dataset == 'cifar10':
                    aux_img = generate_x(3072, lower_i, upper_i)
                else:
                    assert False

                xs.append(aux_img)
                ys.append(target_lst[i])

        return np.array(xs), np.array(ys)


    def __mask_off(self, train_index):
        for index in range(len(train_index)):
            if train_index[index]:
                if random.random() > 0.2:
                    train_index[index] = False


    def __train_robust(self, model, training_mode, train_dataloader, loss_fn, optimizer, device,
        size, group_idx, output_size, old_params, robust_dataloader, lst_poly_lst):
        
        model.train()

        loss, loss1, loss2 = 0.0, 0.0, 0.0
        lamdba1, lambda2 = 1e-3, 1e-3

        idx = group_idx # idx = 1,2,3,4
        size = size * output_size // 2 # size = 10,25

        for batch1, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()

            params = model.named_parameters()
            for name, param in params:
                if 'weight' in name:
                    if 'fc4' in name:
                        for i in range(idx):
                            param.grad.data[output_size*i:output_size*(i+1),size*(i+1):] = 0.0
                    elif 'fc1' not in name:
                        for i in range(idx):
                            param.grad.data[size*i:size*(i+1),size*(i+1):] = 0.0

            optimizer.step()

        if training_mode == 'data_syn' or training_mode == 'continual' or training_mode == 'continual_ext':
            for batch2, (x, y) in enumerate(robust_dataloader):
                x, y = x.to(device), y.to(device)

                # Compute prediction error
                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()

                params = model.named_parameters()
                for name, param in params:
                    if 'weight' in name:
                        if 'fc4' in name:
                            for i in range(idx):
                                param.grad.data[output_size*i:output_size*(i+1),size*(i+1):] = 0.0
                        elif 'fc1' not in name:
                            for i in range(idx):
                                param.grad.data[size*i:size*(i+1),size*(i+1):] = 0.0

                optimizer.step()

        if training_mode == 'continual' or training_mode == 'continual_ext':
            model.fc1.register_forward_hook(get_activation('fc1'))
            model.fc2.register_forward_hook(get_activation('fc2'))
            model.fc3.register_forward_hook(get_activation('fc3'))
            model.fc4.register_forward_hook(get_activation('fc4'))

            for batch3, (x, y) in enumerate(robust_dataloader):
                x, y = x.to(device), y.to(device)
                lst_poly = lst_poly_lst[batch3]

                # Compute prediction error
                pred = model(x)

                for i in range(4):
                    layer = 'fc' + str(i + 1)

                    if i < 3:
                        layer_tensor = activation[layer][:,:size * idx]
                    else:
                        layer_tensor = activation[layer][:,:output_size * idx]

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

            params = model.named_parameters()
            for name, param in params:
                if 'weight' in name:
                    if 'fc4' in name:
                        loss2 = loss2 + ((param.data[:output_size * idx,:size * idx] - old_params[name]) ** 2).sum()
                    elif 'fc1' not in name:
                        loss2 = loss2 + ((param.data[:size * idx,:size * idx] - old_params[name]) ** 2).sum()
                    else:
                        loss2 = loss2 + ((param.data[:size * idx] - old_params[name]) ** 2).sum()
                elif 'bias' in name:
                    if 'fc4' in name:
                        loss2 = loss2 + ((param.data[:output_size * idx] - old_params[name]) ** 2).sum()
                    else:
                        loss2 = loss2 + ((param.data[:size * idx] - old_params[name]) ** 2).sum()

            loss = lamdba1 * loss1 + lambda2 * loss2

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            params = model.named_parameters()
            for name, param in params:
                if 'weight' in name:
                    if 'fc4' in name:
                        for i in range(idx):
                            param.grad.data[output_size*i:output_size*(i+1),size*(i+1):] = 0.0
                    elif 'fc1' not in name:
                        for i in range(idx):
                            param.grad.data[size*i:size*(i+1),size*(i+1):] = 0.0

            optimizer.step()


    def __transfer_model(self, model1, model2, size, group_idx, output_size):
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        idx = group_idx # idx = 1,2,3,4
        size = size * output_size // 2 # size = 10,25

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
                new_data0 = torch.cat((param1.data, dict_params2[name1].data[:output_size * idx,size * idx:]), 1)
                new_data = torch.cat((new_data0, dict_params2[name1].data[output_size * idx:]))
                new_data[:output_size * idx,size * idx:] = 0.0
            elif 'fc4.bias' in name1:
                new_data = torch.cat((param1.data, dict_params2[name1].data[output_size * idx:]))
                
            dict_params2[name1].data.copy_(new_data)
        
        model2.load_state_dict(dict_params2)


    def __write_constr_output_layer(self, y0, y, prob, prev_var_idx):
        prob.write('  x{} - x{} > 0.0\n'.format(prev_var_idx + y, prev_var_idx + y0))
        prob.flush()


    def __verify(self, model, dataset, lower, upper, y0):
        if dataset == 'mnist':
            input_size = 784
        elif dataset == 'cifar10':
            input_size = 3072
        else:
            assert False

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


    def __verify_iteration(self, model, dataset, robust_lst, target_lst, lst_poly_lst, size, num_of_lbls, group_idx, group, device, eps):
        new_lst_poly_lst = []
        if dataset == 'mnist':
            test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
        elif dataset == 'cifar10':
            test_dataset = datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor())
        else:
            assert False

        if dataset == 'cifar10':
            test_dataset.targets = torch.Tensor(test_dataset.targets).type(torch.LongTensor)

        for i in range(0, num_of_lbls):
            if i == 0:
                test_index = test_dataset.targets == 0
            else:
                test_index = test_index | (test_dataset.targets == i)

        test_dataset.data, test_dataset.targets = test_dataset.data[test_index], test_dataset.targets[test_index]
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

        if dataset == 'mnist':
            shape, lower, upper = (1, 784), np.zeros(784), np.ones(784)
        elif dataset == 'cifar10':
            shape, lower, upper = (1, 3072), np.zeros(3072), np.ones(3072)
        else:
            assert False

        formal_model = get_formal_model(model, shape, lower, upper)

        test(model, test_dataloader, nn.CrossEntropyLoss(), device)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        print('robust len = {}'.format(len(robust_lst)))
        pass_cnt, fail_cnt = 0, 0

        for i in range(len(robust_lst)):
            img, target = robust_lst[i], target_lst[i]

            lower_i, upper_i = (img - eps).reshape(-1), (img + eps).reshape(-1)
            lower_i = np.maximum(lower_i, formal_model.lower)
            upper_i = np.minimum(upper_i, formal_model.upper)

            res, lst_poly = self.__verify(formal_model, dataset, lower_i, upper_i, target)
            if res:
                pass_cnt += 1
            else:
                fail_cnt += 1
            new_lst_poly_lst.append(lst_poly)

        print('pass_cnt = {}, fail_cnt = {}, percent = {}'.format(pass_cnt, fail_cnt,
            pass_cnt / len(robust_lst) if len(robust_lst) > 0 else 0))

        for i in range(len(robust_lst)):
            lst_poly_lst[i] = new_lst_poly_lst[i]

        cnt = 0
        if num_of_lbls < 10:
            for data, target in test_dataloader:
                cnt += 1
                if dataset == 'mnist':
                    img = data.numpy().reshape(1, 784)
                elif dataset == 'cifar10':
                    img = data.numpy().reshape(1, 3072)
                else:
                    assert False

                lower_i, upper_i = (img - eps).reshape(-1), (img + eps).reshape(-1)
                lower_i = np.maximum(lower_i, formal_model.lower)
                upper_i = np.minimum(upper_i, formal_model.upper)
                target = target.numpy()[0]

                if target >= group[0]:
                    res, lst_poly = self.__verify(formal_model, dataset, lower_i, upper_i, target)
                    if res:
                        robust_lst.append(img)
                        target_lst.append(target)
                        lst_poly_lst.append(lst_poly)

                        if len(robust_lst) == num_of_lbls * 5:
                            print('Enough robust samples')
                            print(target_lst)
                            break
                    
                    if cnt >= num_of_lbls * 100:
                        assert False


    def solve(self, models, assertion, display=None):
        masked_index_lst = []
        robust_lst, target_lst, lst_poly_lst = [], [], []
        size, device = 10, 'cpu'
        dataset = 'mnist'
        training_mode = 'none_ext'
        already_init = True

        if dataset == 'mnist':
            groups = [[0,1],[2,3],[4,5],[6,7],[8,9]]
            # groups = [[0,1,2,3,4],[5,6,7,8,9]]
            eps = 0.01
        elif dataset == 'cifar10':
            # groups = [[0,1],[2,3],[4,5],[6,7],[8,9]]
            groups = [[0,1,2,3,4],[5,6,7,8,9]]
            eps = 0.001
        else:
            assert False

        for group_idx, group in enumerate(groups):
            print('group = {}'.format(group))
            num_of_lbls = (group_idx + 1) * len(group)

            if dataset == 'mnist':
                train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
                test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
            elif dataset == 'cifar10':
                train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor())
                test_dataset = datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor())
            else:
                assert False

            if dataset == 'cifar10':
                train_dataset.targets = torch.Tensor(train_dataset.targets).type(torch.LongTensor)
                test_dataset.targets = torch.Tensor(test_dataset.targets).type(torch.LongTensor)

            train_index = train_dataset.targets == -1
            if training_mode == 'continual_ext' or training_mode == 'none_ext':
                mask_index = train_dataset.targets == -1

            for idx in group:
                train_index = train_index | (train_dataset.targets == idx)
                if training_mode == 'continual_ext' or training_mode == 'none_ext':
                    mask_index = mask_index | (train_dataset.targets == idx)

                if idx == 0:
                    test_index = test_dataset.targets == 0
                else:
                    test_index = test_index | (test_dataset.targets == idx)

            if training_mode == 'continual_ext' or training_mode == 'none_ext':
                for masked_index in masked_index_lst:
                    train_index = train_index | masked_index

            train_dataset.data, train_dataset.targets = train_dataset.data[train_index], train_dataset.targets[train_index]
            test_dataset.data, test_dataset.targets = test_dataset.data[test_index], test_dataset.targets[test_index]

            if group[0] > 0: old_model = model

            if dataset == 'mnist':
                model = MNISTNet(size, num_of_lbls).to(device)
            elif dataset == 'cifar10':
                model = CIFAR10Net(size, num_of_lbls).to(device)
            else:
                assert False

            if group[0] > 0: self.__transfer_model(old_model, model, size, group_idx, len(group))

            optimizer = optim.SGD(model.parameters(), lr=1e-2)
            if dataset == 'mnist':
                num_of_epochs = 20
            elif dataset == 'cifar10':
                num_of_epochs = 50
            else:
                assert False

            if group[0] == 0:
                if not already_init:
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
                    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
                    
                    best_acc = 0.0

                    for epoch in range(num_of_epochs):
                        print('\n------------- Epoch {} -------------\n'.format(epoch))
                        train(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
                        test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)

                        if test_acc > best_acc:
                            best_acc = test_acc
                            if dataset == 'mnist':
                                save_model(model, 'mnist1.pt')
                            elif dataset == 'cifar10':
                                save_model(model, 'cifar1.pt')
                            else:
                                assert False
            else:
                if dataset == 'mnist':
                    num_samples = 10
                elif dataset == 'cifar10':
                    num_samples = 10
                else:
                    assert False

                if len(robust_lst) > 0 and training_mode != 'none' and training_mode != 'none_ext':
                    aux_robust_lst, aux_target_lst = self.__gen_data(dataset, robust_lst, target_lst, eps, num_samples)
                    print('more train with len = {}'.format(len(aux_robust_lst)))

                    robust_train_x = torch.Tensor(aux_robust_lst.copy()) # transform to torch tensor
                    robust_train_y = torch.Tensor(aux_target_lst.copy()).type(torch.LongTensor)

                    robust_dataset = TensorDataset(robust_train_x, robust_train_y) # create dataset
                    robust_dataloader = DataLoader(robust_dataset, batch_size=num_samples, shuffle=False) # create dataloader
                else:
                    robust_dataloader = None

                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

                best_acc = 0.0
                old_params = {}
                for name, param in old_model.named_parameters():
                    old_params[name] = param.data.clone()

                print('Training data: {}'.format(len(train_dataset.data)))

                if training_mode == 'none':
                    print('\nTrain with new data only!!!\n')
                elif training_mode == 'none_ext':
                    print('\nTrain with new and old data only!!!\n')
                elif training_mode == 'data_syn':
                    print('\nTrain with data synthesis!!!\n')
                elif training_mode == 'continual':
                    print('\nTrain with continual certificate!!!\n')
                elif training_mode == 'continual_ext':
                    print('\nTrain with continual certificate and extra data!!!\n')
                else:
                    assert False

                for epoch in range(num_of_epochs):
                    print('\n------------- Epoch {} -------------\n'.format(epoch))
                    self.__train_robust(model, training_mode, train_dataloader, nn.CrossEntropyLoss(), optimizer, device,
                        size, group_idx, len(group), old_params, robust_dataloader, lst_poly_lst)
                    test_acc = test(model, test_dataloader, nn.CrossEntropyLoss(), device)

                    if test_acc > best_acc:
                        best_acc = test_acc
                        if dataset == 'mnist':
                            file_name = 'mnist' + str(group_idx + 1) + '.pt'
                            save_model(model, file_name)
                        elif dataset == 'cifar10':
                            file_name = 'cifar' + str(group_idx + 1) + '.pt'
                            save_model(model, file_name)
                        else:
                            assert False
            
            if group_idx == 0 and already_init:
                if dataset == 'mnist':
                    model = load_model(MNISTNet, 'mnist2/mnist1.pt', size, num_of_lbls)
                elif dataset == 'cifar10':
                    model = load_model(CIFAR10Net, 'cifar/cifar1.pt', size, num_of_lbls)
                else:
                    assert False
            else:
                if dataset == 'mnist':
                    file_name = 'mnist' + str(group_idx + 1) + '.pt'
                    model = load_model(MNISTNet, file_name, size, num_of_lbls)
                elif dataset == 'cifar10':
                    file_name = 'cifar' + str(group_idx + 1) + '.pt'
                    model = load_model(CIFAR10Net, file_name, size, num_of_lbls)
                else:
                    assert False

            self.__verify_iteration(model, dataset, robust_lst, target_lst, lst_poly_lst, size, num_of_lbls, group_idx, group, device, eps)

            if training_mode == 'continual_ext' or training_mode == 'none_ext':
                self.__mask_off(mask_index)
                masked_index_lst.append(mask_index)
