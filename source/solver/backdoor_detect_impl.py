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


############################# Code adapted from https://github.com/xternalz/WideResNet-pytorch


class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlockWRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlockWRN
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        # out = out.view(-1, self.nChannels)
        out = torch.flatten(out, 1)
        return self.fc(out)

class SubWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(SubWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
    def forward(self, x):
        # out = self.avgpool(x)
        # out = out.view(-1, self.nChannels)
        # out = x.view(-1, self.nChannels)
        out = torch.flatten(x, 1)
        return self.fc(out)


##################################################


############################# Resnet - 18 (https://github.com/Gwinhen/MOTH)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlockRN(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,groups=1, 
        base_width=64, dilation=1, norm_layer=None,
    ):
        super(BasicBlockRN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlockRN only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockRN")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=10, zero_init_residual=False, groups=1, 
        width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockRN):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, previous_dilation, norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class SubResNet(nn.Module):
    def __init__(
        self, block, num_classes=10,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def forward(self, x):
        # x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


##################################################


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


##################################################


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


##################################################


class BackdoorDetectImpl():
    def __transfer_model(self, model, sub_model, dataset):
        params = model.named_parameters()
        sub_params = sub_model.named_parameters()

        dict_params = dict(sub_params)

        # for name, param in params:
        #     if dataset == 'mnist' and ('fc4.weight' in name or 'fc4.bias' in name):
        #         dict_params[name].data.copy_(param.data)
        #     elif (dataset == 'cifar10_inv' or dataset == 'cifar10_sem' or dataset == 'cifar10_tro') \
        #         and ('fc3.weight' in name or 'fc3.bias' in name):
        #         dict_params[name].data.copy_(param.data)

        for name, param in params:
            if dataset == 'mnist' and ('fc4.weight' in name or 'fc4.bias' in name):
                dict_params[name].data.copy_(param.data)
            elif (dataset == 'cifar10_inv' or dataset == 'cifar10_sem' or dataset == 'cifar10_tro') \
                and ('fc3.weight' in name or 'fc3.bias' in name):
                dict_params[name].data.copy_(param.data)
            elif (dataset == 'cifar10_18' or dataset == 'cifar10_34' or dataset == 'cifar10_50') \
                and ('fc.weight' in name or 'fc.bias' in name):
                dict_params[name].data.copy_(param.data)
            elif (dataset == 'WRN-16-1' or dataset == 'WRN-16-2' or dataset == 'WRN-40-1' or \
                dataset == 'WRN-40-2' or dataset == 'WRN-10-1' or dataset == 'WRN-10-2') and \
                ('fc.weight' in name or 'fc.bias' in name):
                dict_params[name].data.copy_(param.data)
            elif dataset == 'cifar' and ('fc3.weight' in name or 'fc3.bias' in name):
                dict_params[name].data.copy_(param.data) # testing with cifar10_bd.pt
        
        sub_model.load_state_dict(dict_params)


    def __generate_trigger(self, model, dataloader, num_of_epochs, size, target, norm, minx = None, maxx = None):
        delta, eps, lamb = torch.zeros(size), 0.001, 1

        for epoch in range(num_of_epochs):
            for batch, (x, y) in enumerate(dataloader):
                delta.requires_grad = True
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                target_tensor = torch.full(y.size(), target)

                pred = model(x_adv)
                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, norm)

                loss.backward()

                grad_data = delta.grad.data
                delta = torch.clamp(delta - eps * grad_data.sign(), -10.0, 10.0).detach()

        return delta


    def __check(self, model, dataloader, delta, target):
        size = len(dataloader.dataset)
        correct = 0

        for batch, (x, y) in enumerate(dataloader):
            x_adv = torch.clamp(torch.add(x, delta), 0.0)
            target_tensor = torch.full(y.size(), target)

            pred = model(x_adv)

            correct += (pred.argmax(1) == target_tensor).type(torch.int).sum().item()

        correct = correct / size * 100
        print('target = {}, test accuracy = {}'.format(target, correct))

        return correct


    def solve(self, model, assertion, display=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_kwargs, test_kwargs = {'batch_size': 100}, {'batch_size': 1000}
        transform = transforms.ToTensor()

        acc_th, ano_th = 40.0, -1.5

        dataset = 'cifar10_34'
        num_of_epochs = 100
        dist_lst, acc_lst = [], []
        norm = 2

        print('dataset =', dataset)

        if dataset == 'mnist':
            file_name = './backdoor_models/mnist_bd.pt'
            last_layer = 'fc3'

            model = load_model(MNISTNet, file_name)
            model.fc3.register_forward_hook(get_activation(last_layer))
        
            sub_model = SubMNISTNet()
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)

            size_input, size_last = (28, 28), 10

        elif dataset == 'cifar10_inv' or dataset == 'cifar10_sem' or dataset == 'cifar10_tro':
            if dataset == 'cifar10_inv':
                file_name = './backdoor_models/cifar10_bd_inv.pt'
            elif dataset == 'cifar10_sem':
                file_name = './backdoor_models/cifar10_bd_sem.pt'
            elif dataset == 'cifar10_tro':
                file_name = './backdoor_models/cifar10_bd_tro.pt'

            last_layer = 'fc2'

            model = load_model(CIFAR10Net, file_name)
            model.fc2.register_forward_hook(get_activation(last_layer))
        
            sub_model = SubCIFAR10Net()

            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

            size_input, size_last = (3, 32, 32), 512

        elif dataset == 'cifar10_18' or dataset == 'cifar10_34' or dataset == 'cifar10_50':

            def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
                model = ResNet(block, layers, **kwargs)
                if pretrained:
                    if file_name.endswith('tar'):
                        checkpoint = torch.load(file_name, map_location='cpu')
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        state_dict = torch.load(file_name, map_location=device)
                        model.load_state_dict(state_dict)
                return model

            last_layer = 'avgpool'

            if dataset == 'cifar10_18':
                file_name = './backdoor_models/cifar10_resnet18_bd.pt'
                model = _resnet("resnet18", BasicBlockRN, [2, 2, 2, 2], pretrained=True, progress=True, device="cpu")
                model.avgpool.register_forward_hook(get_activation(last_layer))
                sub_model = SubResNet(BasicBlockRN)
            
            elif dataset == 'cifar10_34':
                file_name = './backdoor_models/cifar10_resnet34_bd.pth.tar'
                model = _resnet("resnet34", BasicBlockRN, [3, 4, 6, 3], pretrained=True, progress=True, device="cpu")
                model.avgpool.register_forward_hook(get_activation(last_layer))
                sub_model = SubResNet(BasicBlockRN)
            
            elif dataset == 'cifar10_50':
                file_name = './backdoor_models/cifar10_wrn_bd.pth.tar'
                model = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True, device="cpu")
                model.avgpool.register_forward_hook(get_activation(last_layer))
                sub_model = SubResNet(Bottleneck)
                
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

            size_input, size_last = (3, 32, 32), 512
        
        elif dataset == 'WRN-16-1' or dataset == 'WRN-16-2' or dataset == 'WRN-40-1' or \
            dataset == 'WRN-40-2' or dataset == 'WRN-10-1' or dataset == 'WRN-10-2':
            
            last_layer = 'avgpool'

            if dataset == 'WRN-16-1':
                file_name = './backdoor_models/WRN-16-1-fourCornerTrigger.pth.tar'
            elif dataset == 'WRN-16-2':
                file_name = './backdoor_models/WRN-16-2-fourCornerTrigger.pth.tar'
            elif dataset == 'WRN-40-1':
                file_name = './backdoor_models/WRN-40-1-fourCornerTrigger.pth.tar'
            elif dataset == 'WRN-40-2':
                file_name = './backdoor_models/WRN-40-2-fourCornerTrigger.pth.tar'
            elif dataset == 'WRN-10-1':
                file_name = './backdoor_models/WRN-10-1-fourCornerTrigger.pth.tar'
            elif dataset == 'WRN-10-2':
                file_name = './backdoor_models/WRN-10-2-fourCornerTrigger.pth.tar'
            
            def _wresnet(depth, num_classes, widen_factor, dropRate, pretrained=True, **kwargs):
                model = WideResNet(depth, num_classes, widen_factor, dropRate)

                if pretrained:
                    checkpoint = torch.load(file_name, map_location='cpu')
                    model.load_state_dict(checkpoint['state_dict'])
                return model

            if dataset=='WRN-16-1':
                model = _wresnet(depth=16, num_classes=10, widen_factor=1, dropRate=0)
                sub_model = SubWideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0)
                size_input, size_last = (3, 32, 32), 64
            elif dataset=='WRN-16-2':
                model = _wresnet(depth=16, num_classes=10, widen_factor=2, dropRate=0)
                sub_model = SubWideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0)
                size_input, size_last = (3, 32, 32), 128
            elif dataset=='WRN-40-1':
                model = _wresnet(depth=40, num_classes=10, widen_factor=1, dropRate=0)
                sub_model = SubWideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0)
                size_input, size_last = (3, 32, 32), 64
            elif dataset=='WRN-40-2':
                model = _wresnet(depth=40, num_classes=10, widen_factor=2, dropRate=0)
                sub_model = SubWideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0)
                size_input, size_last = (3, 32, 32), 128
            elif dataset == 'WRN-10-1':
                model = _wresnet(depth=10, num_classes=10, widen_factor=1, dropRate=0)
                sub_model = SubWideResNet(depth=10, num_classes=10, widen_factor=1, dropRate=0)
                size_input, size_last = (3, 32, 32), 64
            elif dataset == 'WRN-10-2':
                model = _wresnet(depth=10, num_classes=10, widen_factor=2, dropRate=0)
                sub_model = SubWideResNet(depth=10, num_classes=10, widen_factor=2, dropRate=0)
                size_input, size_last = (3, 32, 32), 128
            else:
                raise NotImplementedError

            model.avgpool.register_forward_hook(get_activation(last_layer))
            
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        
        else:
            assert False

        self.__transfer_model(model, sub_model, dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        last_layer_test_dataset = []
        for batch, (x, y) in enumerate(test_dataloader):
            model(x)
            if dataset == 'mnist' or dataset == 'cifar10_inv' or dataset == 'cifar10_sem' or dataset == 'cifar10_tro':
                last_layer_test_dataset.extend(F.relu(activation[last_layer]).detach().numpy())
            elif dataset == 'WRN-16-1' or dataset == 'WRN-16-2' or dataset == 'WRN-40-1' or \
                dataset == 'WRN-40-2' or dataset == 'WRN-10-1' or dataset == 'WRN-10-2' or \
                dataset == 'cifar10_18' or dataset == 'cifar10_34' or dataset == 'cifar10_50':
                last_layer_test_dataset.extend(torch.flatten(activation[last_layer], 1).detach().numpy())

        last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)), torch.Tensor(np.array(test_dataset.targets))) # create dataset
        last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **test_kwargs) # create dataloader

        # dataloader = test_dataloader
        dataloader = last_layer_test_dataloader

        for target in range(10):
            # delta = self.__generate_trigger(model, dataloader, num_of_epochs, size_input, target, norm, 0.0, 1.0)
            delta = self.__generate_trigger(sub_model, dataloader, num_of_epochs, size_last, target, norm, 0.0)
            delta = torch.where(abs(delta) < 0.1, delta - delta, delta)

            print('\n###############################\n')

            # acc = self.__check(model, dataloader, delta, target)
            acc = self.__check(sub_model, dataloader, delta, target)
            dist = torch.norm(delta, 0)
            print('\ntarget = {}, delta = {}, dist = {}\n'.format(target, delta[:10], dist))

            acc_lst.append(acc)
            dist_lst.append(dist.detach().item())

        print('\n###############################\n')

        acc_lst = np.array(acc_lst)
        print('acc_lst = {}'.format(acc_lst))

        dist_lst = np.array(dist_lst)
        print('dist_lst = {}'.format(dist_lst))

        med = statistics.median(dist_lst)
        print('med = {}'.format(med))
        
        dev_lst = abs(dist_lst - med)
        print('dev_lst = {}'.format(dev_lst))
        
        mad = statistics.median(dev_lst)
        print('mad = {}'.format(mad))
        
        ano_lst = (dist_lst - med) / mad
        print('ano_lst = {}'.format(ano_lst))

        print('\n###############################\n')

        for target in range(10):
            if acc_lst[target] >= acc_th and ano_lst[target] <= ano_th:
                print('Detect backdoor at target = {}'.format(target))
