import torch
import random
import numpy as np

from utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# Define model
class NewNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 50),
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
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ContinualImpl():
    def loss_fn0(self, pred, x, y):
        ce_loss_fn = nn.CrossEntropyLoss()
        return ce_loss_fn(pred, y)


    def loss_fn1(self, pred, x, y):
        loss1 = self.loss_fn0(pred, x, y)

        if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
            -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
            if torch.take(pred, torch.tensor([0])) <= 3.9911:
                loss2 = 0.0
            else:
                loss2 = torch.take(pred, torch.tensor([0])) - 3.9911
        else:
            loss2 = 0.0

        return loss1 + loss2


    def loss_fn2(self, pred, x, y):
        loss1 = self.loss_fn0(pred, x, y)

        if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
            -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
            if torch.argmax(pred) != 0:
                loss2 = 0.0
            else:
                loss2 = torch.take(pred, torch.tensor([0]))
        else:
            loss2 = 0.0

        return loss1 + loss2


    def loss_fn3(self, pred, x, y):
        loss1 = self.loss_fn0(pred, x, y)

        if -0.3035 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= -0.2986 and \
            -0.0095 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.0095 and \
            0.4934 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
            0.3 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= 0.5:
            if torch.argmin(pred) != 0:
                loss2 = 0.0
            else:
                loss2 = -torch.take(pred, torch.tensor([0]))
        else:
            loss2 = 0.0

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

            if batch % 10 == 0:
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


    def solve(self, model, assertion, display=None):
        # lower, upper = model.lower, model.upper
        # bound: [(-0.3284,0.6799), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]
        lower = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        print("lower = {}".format(lower))
        print("upper = {}".format(upper))
        xs, ys = [], []

        for i in range(10000):
            x = generate_x(len(lower), lower, upper)
            y = np.argmin(model.apply(x).reshape(-1))

            xs.append(x)
            ys.append(y)

        xs, ys = np.array(xs), np.array(ys)

        print("xs.shape = {}".format(xs.shape))
        print("ys.shape = {}".format(ys.shape))

        train_x, train_y = xs[:8000], ys[:8000]
        test_x, test_y = xs[8000:], ys[8000:]

        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).long()

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).long()

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        device = 'cpu'
        new_model = NewNetwork().to(device)

        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-4)

        num_of_epochs = 1
        num_of_properties = 3

        loss_fn_arr = [self.loss_fn0, self.loss_fn1, self.loss_fn2, self.loss_fn3]
        
        for i in range(num_of_properties + 1):
            if i == 0:
                print('\n------------- Initial -------------\n')
            else:
                print('\n------------- Property {} -------------\n'.format(i))

            for j in range(num_of_epochs):
                print('\n------------- Epoch {} -------------\n'.format(j))
                self.train(new_model, train_dataloader, loss_fn_arr[i], optimizer, device)
                self.test(new_model, test_dataloader, loss_fn_arr[i], device)

        return None