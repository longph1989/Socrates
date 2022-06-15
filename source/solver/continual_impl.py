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

# Define model
class NewNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ContinualImpl():
    def __loss_fn0(self, pred, x, y):
        loss = nn.CrossEntropyLoss()
        return loss(pred, y)


    # loss function to keep prop 3
    def __loss_fn3(self, pred, x, y):
        loss1 = self.__loss_fn0(pred, x, y)
        loss2 = 0.0

        if -0.3035 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= -0.2986 and \
            -0.0095 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.0095 and \
            0.4934 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= 0.5:
            if torch.argmax(pred) == 0:
                # reduce the distance between the value at index 0 and other so it is no longer the max
                for i in range(1,5):
                    loss2 += 100 * (torch.take(pred, torch.tensor([0])) - torch.take(pred, torch.tensor([i])))

        return loss1 + loss2


    def __train(self, new_model, dataloader, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        new_model.train()

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = new_model(x)
            loss = loss_fn(pred, x, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def __test(self, new_model, dataloader, loss_fn, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        new_model.eval()
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = new_model(x)
                test_loss += loss_fn(pred, x, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def __run(self, model, idx, lst_poly):
        print('idx =', idx, 'lw =', lst_poly[-1].lw)
        print('idx =', idx, 'up =', lst_poly[-1].up)

        if idx == len(model.layers):
            poly_out = lst_poly[idx]
            return poly_out
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)

            lw_next = poly_next.lw
            up_next = poly_next.up

            if np.any(up_next < lw_next):
                assert False # unreachable states

            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def __verify(self, model, lower, upper, prop):
        x0_poly = Poly()

        x0_poly.lw, x0_poly.up = lower, upper
        # just let x0_poly.le and x0_poly.ge is None
        x0_poly.shape = model.shape

        lst_poly = [x0_poly]
        poly_out = self.__run(model, 0, lst_poly)

        if prop == 3:
            out_lw = poly_out.lw.copy()
            out_lw[0] = poly_out.up[0]

            if np.argmax(out_lw) == 0:
                print(out_lw)
                assert False
        elif prop == 4:
            out_lw = poly_out.lw.copy()
            out_lw[0] = poly_out.up[0]

            if np.argmax(out_lw) == 0: assert False


    def __gen_train_data_rec(self, model, dims, xs, ys, x, idx, ln, x1, x2):
        if idx == ln:
            # output based on the model
            new_x = x.copy()
            y = model.apply(np.array(new_x)).reshape(-1)

            # new dims for x
            new_x.append((x1 - 2) / 4)
            new_x.append((x2 - 4) / 8)
            new_x = np.array(new_x)

            xs.append(new_x)
            ys.append(y)
        else:
            for val in dims[idx]:
                x.append(val)
                self.__gen_train_data_rec(model, dims, xs, ys, x, idx + 1, ln, x1, x2)
                x.pop()


    def __gen_train_data(self, models, lower, upper, aux=False, skip=[]):
        xs, ys, dims = [], [], []
        for i in range(len(lower)):
            dims.append(np.linspace(lower[i], upper[i], num=5))

        for x1 in range(5):
            for x2 in range(9):
                if x1 in skip: continue
                model = models[x1][x2]
                self.__gen_train_data_rec(model, dims, xs, ys, [], 0, len(lower), x1, x2)

        xs = np.array(xs)
        ys = np.argmin(ys, axis=1).reshape(-1) # get label

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


    def __gen_test_data(self, models, lower, upper):
        xs, ys = [], []

        for i in range(100000):
            x = list(generate_x(5, lower, upper))
            x1, x2 = random.randint(0, 4), random.randint(0, 8)

            y = models[x1][x2].apply(np.array(x)).reshape(-1)
            x.append((x1 - 2) / 4)
            x.append((x2 - 4) / 8)

            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.argmin(ys, axis=1).reshape(-1) # get label

        unique, counts = np.unique(ys, return_counts=True)
        print('Test dist = {}'.format(dict(zip(unique, counts))))

        return xs, ys


    def __train_new_model(self, new_model, device, train_x, train_y, test_x, test_y, loss_fn):
        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)

        num_of_epochs = 200

        for i in range(num_of_epochs):
            print('\n------------- Epoch {} -------------\n'.format(i))
            self.__train(new_model, train_dataloader, loss_fn, optimizer, device)
            self.__test(new_model, train_dataloader, loss_fn, device)
            self.__test(new_model, test_dataloader, loss_fn, device)

            # scheduler.step()

        return new_model


    def __save_model(self, model, name):
        torch.save(model.state_dict(), name)


    def __load_model(self, name):
        load_model = NewNetwork()
        load_model.load_state_dict(torch.load(name))

        return load_model


    def __prop3_test(self, model):
        lower = np.array([-0.3035, -0.0095, 0.4934, 0.3, 0.3, -0.25, -0.5])
        upper = np.array([-0.2986, 0.0095, 0.5, 0.5, 0.5, 0.5, 0.5])

        threshold, alpha, beta, delta = 0.99, 0.01, 0.01, 0.005

        p0 = threshold + delta
        p1 = threshold - delta

        h0 = beta / (1 - alpha)
        h1 = (1 - beta) / alpha

        pr, no = 1, 0

        with torch.no_grad():
            for i in range(int(1e6)):
                x = torch.Tensor(generate_x(7, lower, upper))
                pred = model(x).numpy()

                no = no + 1

                if np.argmax(pred) != 0:
                    pr = pr * p1 / p0
                else:
                    assert False
                    pr = pr * (1 - p1) / (1 - p0)

                # if pr <= h0:
                #     print('Accept H0. The assertion is satisfied with p >= {} after {} tests.'.format(p0, no))
                #     break
                # elif pr >= h1:
                #     print('Accept H1. The assertion is satisfied with p <= {} after {} tests.'.format(p1, no))
                #     break
        print(no)


    def __get_layers(self, model):
        layers, params = list(), list(model.named_parameters())

        for i in range(len(params)):
            name, param = params[i]
            if 'weight' in name:
                weight = np.array(param.data)
                print(name)
                print(weight.shape)
            elif 'bias' in name:
                bias = np.array(param.data)
                print(name)
                print(bias.shape)

                layers.append(Linear(weight, bias, None))
                if i < len(params) - 1: # last layer
                    layers.append(Function('relu', None))

        return layers


    def __get_formal_model(self, model, lower, upper):
        shape = np.array([1,7])
        lower, upper = lower.copy(), upper.copy()
        layers = self.__get_layers(model)
        
        return Model(shape, lower, upper, layers, None)


    def solve(self, models, assertion, display=None):
        # data from normal bounds

        lower0 = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper0 = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        # train_x0, train_y0 = self.__gen_train_data(models, lower0, upper0, aux=True)
        # test_x, test_y = self.__gen_test_data(models, lower0, upper0)

        # # data from condition 3 bounds

        lower3 = np.array([-0.3035, -0.0095, 0.4934, 0.3, 0.3])
        upper3 = np.array([-0.2986, 0.0095, 0.5, 0.5, 0.5])

        # train_x3, train_y3 = self.__gen_train_data(models, lower3, upper3, skip=[0])

        # device = 'cpu'
        # new_model0 = NewNetwork().to(device)

        # train_x = np.concatenate((train_x0, train_x3), axis=0)
        # train_y = np.concatenate((train_y0, train_y3), axis=0)
        # new_model1 = self.__train_new_model(new_model0, device, train_x, train_y, test_x, test_y, self.__loss_fn3)
        # self.__save_model(new_model1, "model1b.pt")

        new_model1 = self.__load_model("model1b.pt")
        print('finish model 1')

        self.__prop3_test(new_model1)

        formal_lower0, formal_upper0 = list(lower0.copy()), list(upper0.copy())
        formal_lower0.extend([-0.5, -0.5])
        formal_upper0.extend([0.5, 0.5])

        formal_lower3, formal_upper3 = list(lower3.copy()), list(upper3.copy())
        formal_lower3.extend([-0.25, -0.5])
        formal_upper3.extend([0.5, 0.5])

        formal_model = self.__get_formal_model(new_model1, np.array(formal_lower0), np.array(formal_upper0))
        self.__verify(formal_model, np.array(formal_lower3), np.array(formal_upper3), 3)

        # data from condition 4 bounds

        # lower4 = np.array([-0.3035, -0.0095, 0.0, 0.3182, 0.0833])
        # upper4 = np.array([-0.2986, 0.0095, 0.0, 0.5, 0.1667])

        # train_x4, train_y4 = self.__gen_train_data(models, lower4, upper4, skip=[0])
        
        # train_x, train_y = train_x4, train_y4
        
        # random_x0 = random.sample(range(len(train_x0)), len(train_x0) // 5)
        # random_x3 = random.sample(range(len(train_x3)), len(train_x3) // 5)

        # aux_train_x0, aux_train_y0 = train_x0[random_x0], train_y0[random_x0]
        # aux_train_x3, aux_train_y3 = train_x3[random_x3], train_y3[random_x3]
        
        # train_x = np.concatenate((train_x, aux_train_x0), axis=0)
        # train_y = np.concatenate((train_y, aux_train_y0), axis=0)
        # train_x = np.concatenate((train_x, aux_train_x3), axis=0)
        # train_y = np.concatenate((train_y, aux_train_y3), axis=0)

        # new_model2 = self.__train_new_model(new_model1, device, train_x, train_y, test_x, test_y, self.__loss_fn3)
        # print('finish model 2')
        
        # self.__prop3_test(new_model2)

        # formal_model = self.__get_formal_model(new_model2, np.array(formal_lower0), np.array(formal_upper0))
        # self.__verify(formal_model, np.array(formal_lower3), np.array(formal_upper3), 3)


class MNISTNet(nn.Module):
    def __init__(self, lbl):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, lbl)

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
        self.cnt = 0
        self.pos = 0

    def __train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    def __test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def __save_model(self, model, name):
        torch.save(model.state_dict(), name)


    def __load_model(self, name):
        load_model = MNISTNet1()
        load_model.load_state_dict(torch.load(name))

        return load_model


    def __transfer_model(self, model1, model2):
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 == 'fc4.weight' or name1 == 'fc4.bias': continue
            dict_params2[name1].data.copy_(param1.data)

        model2.load_state_dict(dict_params2)


    def __print_model(self, model):
        for name, param in model.named_parameters():
            print(name)
            print(param.data)


    def __mask_off(self, train_index):
        for index in range(len(train_index)):
            if train_index[index]:
                if random.random() > 0.2:
                    train_index[index] = False


    def __train_iteration(self):
        device = torch.device("cpu")

        train_kwargs = {'batch_size': 100}
        test_kwargs = {'batch_size': 1000}

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        for lbl in range(2, 11, 2):
            print(lbl)

            train = datasets.MNIST('../data', train=True, download=True, transform=transform)
            test = datasets.MNIST('../data', train=False, transform=transform)

            if lbl == 2:
                train_index, test_index = train.targets == 0, test.targets == 0
            
            for i in range(lbl - 2, lbl):
                if i == 0: continue
                train_index = train_index | (train.targets == i)
                test_index = test_index | (test.targets == i)

            train.data, train.targets = train.data[train_index], train.targets[train_index]
            test.data, test.targets = test.data[test_index], test.targets[test_index]

            train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(test, **test_kwargs)

            if lbl > 2: old_model = model
            model = MNISTNet(lbl).to(device)
            if lbl > 2: self.__transfer_model(old_model, model)

            optimizer = optim.SGD(model.parameters(), lr=1e-3)

            for epoch in range(lbl * 10):
                self.__train(model, device, train_loader, optimizer, epoch)
                self.__test(model, device, test_loader)

            if lbl == 2:
                self.__save_model(model, 'mnist1.pt')
            elif lbl == 4:
                self.__save_model(model, 'mnist2.pt')
            elif lbl == 6:
                self.__save_model(model, 'mnist3.pt')
            elif lbl == 8:
                self.__save_model(model, 'mnist4.pt')
            elif lbl == 10:
                self.__save_model(model, 'mnist5.pt')

            self.__mask_off(train_index)


    def solve(self, models, assertion, display=None):
        self.__train_iteration()
        
        return None 
