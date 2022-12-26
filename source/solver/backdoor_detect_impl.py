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
import statistics



class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

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


class SubMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


class CIFAR10Net(nn.Module):
    # from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # output: 64 x 16 x 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # output: 128 x 8 x 8

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # output: 256 x 4 x 4

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = x
        return output


class SubCIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc3(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


class BackdoorDetectImpl():
    def __transfer_model(self, model, sub_model, dataset):
        params = model.named_parameters()
        sub_params = sub_model.named_parameters()

        dict_params = dict(sub_params)

        for name, param in params:
            if dataset == 'mnist' and ('fc4.weight' in name or 'fc4.bias' in name):
                dict_params[name].data.copy_(param.data)
            elif dataset == 'cifar10' and ('fc3.weight' in name or 'fc3.bias' in name):
                dict_params[name].data.copy_(param.data)
        
        sub_model.load_state_dict(dict_params)


    def __generate_trigger(self, model, dataloader, num_of_epochs, size, target, minx = None, maxx = None):
        delta, eps, lamb = torch.zeros(size), 0.01, 1

        for epoch in range(num_of_epochs):
            for batch, (x, y) in enumerate(dataloader):
                delta.requires_grad = True
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                target_tensor = torch.full(y.size(), target)

                pred = model(x_adv)
                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, 2)

                loss.backward()

                grad_data = delta.grad.data
                delta = torch.clamp(delta - eps * grad_data.sign(), -10.0, 10.0).detach()

        return delta


    def solve(self, model, assertion, display=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_kwargs, test_kwargs = {'batch_size': 100}, {'batch_size': 1000}
        transform = transforms.ToTensor()

        dataset = 'cifar10'

        if dataset == 'mnist':
            file_name = './backdoor_models/mnist_bd.pt'
            model = load_model(MNISTNet, file_name)
            model.fc3.register_forward_hook(get_activation('fc3'))
        
            sub_model = SubMNISTNet()
            self.__transfer_model(model, sub_model, dataset)

            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

            last_layer_test_dataset = []
            for batch, (x, y) in enumerate(test_dataloader):
                model(x)
                last_layer_test_dataset.extend(F.relu(activation['fc3']).detach().numpy())

            last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)), torch.Tensor(np.array(test_dataset.targets))) # create dataset
            last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **test_kwargs) # create dataloader

            num_of_epochs = 20
            dist_lst = []

            for i in range(0, 10):
                delta = self.__generate_trigger(model, test_dataloader, num_of_epochs, (28, 28), i, 0.0, 1.0)
                # delta = self.__generate_trigger(sub_model, last_layer_test_dataloader, num_of_epochs, 10, i, 0.0)
                dist = torch.norm(delta, 2)
                print('i = {}, delta = {}, d = {}'.format(i, delta, dist))
                dist_lst.append(dist.detach().item())
        elif dataset == 'cifar10':
            file_name = './backdoor_models/cifar10_bd.pt'
            model = load_model(CIFAR10Net, file_name)
            model.fc2.register_forward_hook(get_activation('fc2'))
        
            sub_model = SubCIFAR10Net()
            self.__transfer_model(model, sub_model, dataset)

            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

            last_layer_test_dataset = []
            for batch, (x, y) in enumerate(test_dataloader):
                model(x)
                last_layer_test_dataset.extend(F.relu(activation['fc2']).detach().numpy())

            last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)), torch.Tensor(np.array(test_dataset.targets))) # create dataset
            last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **test_kwargs) # create dataloader

            num_of_epochs = 1
            dist_lst = []

            for i in range(0, 10):
                delta = self.__generate_trigger(model, test_dataloader, num_of_epochs, (3, 32, 32), i, 0.0, 1.0)
                # delta = self.__generate_trigger(sub_model, last_layer_test_dataloader, num_of_epochs, 512, i, 0.0)
                dist = torch.norm(delta, 2)
                print('i = {}, delta = {}, d = {}'.format(i, delta, dist))
                dist_lst.append(dist.detach().item())

        dist_lst = np.array(dist_lst)
        print('dist_lst = {}'.format(dist_lst))
        med = statistics.median(dist_lst)
        print('med = {}'.format(med))
        dev_lst = abs(dist_lst - med)
        print('dev_lst = {}'.format(dev_lst))
        mad = statistics.median(dev_lst)
        print('mad = {}'.format(mad))
        print('res = {}'.format((dist_lst - med) / mad))
