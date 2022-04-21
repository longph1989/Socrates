import torch
import random
import numpy as np

from utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


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
    def loss_fn0(self, pred, x, y):
        loss = nn.CrossEntropyLoss()
        return loss(pred, y)


    def loss_fn1(self, pred, x, y):
        pass
        # loss1 = self.loss_fn0(pred, x, y)

        # if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
        #     -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
        #     -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
        #     0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
        #     -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
        #     if torch.take(pred, torch.tensor([0])) <= 3.9911:
        #         loss2 = 0.0
        #     else:
        #         loss2 = torch.take(pred, torch.tensor([0])) - 3.9911
        # else:
        #     loss2 = 0.0

        # return loss1 + loss2


    def loss_fn2(self, pred, x, y):
        loss1 = self.loss_fn0(pred, x, y)
        loss2 = 0.0

        if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
            -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
            if torch.argmin(pred) == 0:
                # reduce the distance between the value at index 0 and other so it is no longer the min
                for i in range(1,5):
                    loss2 += torch.take(pred, torch.tensor([i])) - torch.take(pred, torch.tensor([0]))

        return loss1 + loss2


    def loss_fn3(self, pred, x, y):
        loss1 = self.loss_fn0(pred, x, y)
        loss2 = 0.0

        if -0.3035 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= -0.2986 and \
            -0.0095 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.0095 and \
            0.4934 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= 0.5:
            if torch.argmax(pred) == 0:
                # reduce the value at index 0 so it is no longer the max
                loss2 = torch.take(pred, torch.tensor([0]))

        return loss1 + loss2


    def train(self, new_model, dataloader, loss_fn, optimizer, device):
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


    def test(self, new_model, dataloader, loss_fn, device):
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


    def __gen_data_rec(self, model, dims, xs, ys, x, idx, ln, x1, x2):
        if idx == ln:
            # output based on the model
            new_x = x.copy()
            y = model.apply(np.array(new_x)).reshape(-1)

            # new dims for x
            new_x.append((x1 - 2) / 5)
            new_x.append((x2 - 4) / 9)
            new_x = np.array(new_x)

            xs.append(new_x)
            ys.append(y)
        else:
            for val in dims[idx]:
                x.append(val)
                self.__gen_data_rec(model, dims, xs, ys, x, idx + 1, ln, x1, x2)
                x.pop()


    def __gen_data(self, models):
        # bound: [(-0.3284,0.6799), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]

        lower = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        xs, ys = [], []

        dims = []
        for i in range(len(lower)):
            dims.append(np.linspace(lower[i], upper[i], num=5))

        for x1 in range(5):
            for x2 in range(9):
                model = models[x1][x2]
                self.__gen_data_rec(model, dims, xs, ys, [], 0, len(lower), x1, x2)
        
        xs, ys = np.array(xs), np.array(ys)

        all_data = np.concatenate((xs, ys), axis=1)
        print(all_data.shape) # should be 5 ^ 5 * 45 = 140,625

        # randomize all data
        np.random.shuffle(all_data)

        xs = all_data[:,:7]
        ys = np.argmin(all_data[:,-5:], axis=1).reshape(-1) # get label

        train_x, train_y = xs[:100000], ys[:100000]
        test_x, test_y = xs[100000:], ys[100000:]

        unique, counts = np.unique(train_y, return_counts=True)
        print('Train dist = {}'.format(dict(zip(unique, counts))))

        unique, counts = np.unique(test_y, return_counts=True)
        print('Test dist = {}'.format(dict(zip(unique, counts))))

        # add auxiliary data to training set
        aux_x, aux_y = [], []

        for i in range(100000):
            if train_y[i] != 0:
                for j in range(9):
                    aux_x.append(train_x[i])
                    aux_y.append(train_y[i])

        train_x = np.concatenate((train_x, np.array(aux_x)), axis=0)
        train_y = np.concatenate((train_y, np.array(aux_y)), axis=0)

        unique, counts = np.unique(train_y, return_counts=True)
        print('Aux train dist = {}'.format(dict(zip(unique, counts))))

        return train_x, train_y, test_x, test_y


    def __train_new_model(self, train_x, train_y, test_x, test_y, loss_fn):
        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        device = 'cpu'
        new_model = NewNetwork().to(device)

        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)

        num_of_epochs = 200

        for i in range(num_of_epochs):
            print('\n------------- Epoch {} -------------\n'.format(i))
            self.train(new_model, train_dataloader, loss_fn, optimizer, device)
            self.test(new_model, train_dataloader, loss_fn, device)
            self.test(new_model, test_dataloader, loss_fn, device)

            # scheduler.step()

        return new_model


    def __save_model(self, model, name):
        torch.save(model.state_dict(), name)


    def __load_model(self, name):
        load_model = NewNetwork()
        load_model.load_state_dict(torch.load(name))

        return load_model


    def __hyp2_test(self, model):
        lower = np.array([0.6, -0.5, -0.5, 0.45, -0.5, -0.4, -0.4444])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, -0.45, 0.4, 0.4444])

        threshold, alpha, beta, delta = 0.99, 0.05, 0.05, 0.005

        p0 = threshold + delta
        p1 = threshold - delta

        h0 = beta / (1 - alpha)
        h1 = (1 - beta) / alpha

        pr, no = 1, 0

        with torch.no_grad():
            while True:
                x = torch.Tensor(generate_x(7, lower, upper))
                pred = model(x).numpy()

                no = no + 1

                if np.argmin(pred) != 0:
                    pr = pr * p1 / p0
                else:
                    pr = pr * (1 - p1) / (1 - p0)

                if pr <= h0:
                    print('Accept H0. The assertion is satisfied with p >= {} after {} tests.'.format(p0, no))
                    break
                elif pr >= h1:
                    print('Accept H1. The assertion is satisfied with p <= {} after {} tests.'.format(p1, no))
                    break


    def __hyp3_test(self, model):
        lower = np.array([-0.3035, -0.0095, 0.4934, 0.3, 0.3, -0.4, -0.4444])
        upper = np.array([-0.2986, 0.0095, 0.5, 0.5, 0.5, 0.4, 0.4444])

        threshold, alpha, beta, delta = 0.99, 0.05, 0.05, 0.005

        p0 = threshold + delta
        p1 = threshold - delta

        h0 = beta / (1 - alpha)
        h1 = (1 - beta) / alpha

        pr, no = 1, 0

        with torch.no_grad():
            while True:
                x = torch.Tensor(generate_x(7, lower, upper))
                pred = model(x).numpy()

                no = no + 1

                if np.argmax(pred) != 0:
                    pr = pr * p1 / p0
                else:
                    pr = pr * (1 - p1) / (1 - p0)

                if pr <= h0:
                    print('Accept H0. The assertion is satisfied with p >= {} after {} tests.'.format(p0, no))
                    break
                elif pr >= h1:
                    print('Accept H1. The assertion is satisfied with p <= {} after {} tests.'.format(p1, no))
                    break


    def solve(self, models, assertion, display=None):
        # train_x, train_y, test_x, test_y = self.__gen_data(models)

        # loss_fn_arr = [self.loss_fn0, self.loss_fn1, self.loss_fn2, self.loss_fn3]
        # new_model = self.__train_new_model(train_x, train_y, test_x, test_y, loss_fn_arr[3])

        # self.__save_model(new_model, 'new_model.pt')
        new_model = self.__load_model('new_model3.pt')

        # self.__hyp2_test(new_model)
        self.__hyp3_test(new_model)
