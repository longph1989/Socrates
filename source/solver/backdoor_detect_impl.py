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


class BackdoorDetectImpl():
    def __transfer_model(self, model, sub_model):
        params = model.named_parameters()
        sub_params = sub_model.named_parameters()

        dict_params = dict(sub_params)

        for name, param in params:
            if 'fc4.weight' in name or 'fc4.bias' in name:
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

        file_name = './backdoor_models/mnist_bd.pt'
        model = load_model(MNISTNet, file_name)
        model.fc3.register_forward_hook(get_activation('fc3'))
        
        sub_model = SubMNISTNet()
        self.__transfer_model(model, sub_model)

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        last_layer_test_dataset = []
        for batch, (x, y) in enumerate(test_dataloader):
            model(x)
            last_layer_test_dataset.extend(F.relu(activation['fc3']).detach().numpy())

        # last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)), test_dataset.targets) # create dataset
        # last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **test_kwargs) # create dataloader

        aa = np.array(last_layer_test_dataset)
        print(aa.shape)
        print(type(aa))
        for i in range(0, 10):
            idx = test_dataset.targets == i
            cc = aa[idx.numpy()]
            print(type(cc))
            print(cc.shape)
            print('i = {}, cc = {}'.format(i, np.average(cc, axis=0)))

        assert False

        num_of_epochs = 100

        for i in range(0, 10):
            # delta = self.__generate_trigger(model, test_dataloader, num_of_epochs, (28, 28), i, 0.0, 1.0)
            delta = self.__generate_trigger(sub_model, last_layer_test_dataloader, num_of_epochs, 10, i, 0.0)
            print('i = {}, delta = {}, d = {}'.format(i, delta, torch.norm(delta, 2)))

        # assert False