import torch
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
    def solve(self, model, assertion, display=None):
        lower, upper = model.lower, model.upper

        print("lower = {}".format(lower))
        print("upper = {}".format(upper))
        xs, ys = [], []

        for i in range(1000):
            x = generate_x(len(lower), lower, upper)
            y = np.argmin(model.apply(x).reshape(-1))

            # print("x = {}".format(x))
            # print("y = {}".format(y))

            xs.append(x)
            ys.append(y)

        xs, ys = np.array(xs), np.array(ys)
        print(ys)

        print("xs.shape = {}".format(xs.shape))
        print("ys.shape = {}".format(ys.shape))

        tensor_x = torch.Tensor(xs) # transform to torch tensor
        tensor_y = torch.Tensor(ys).long()

        dataset = TensorDataset(tensor_x, tensor_y) # create datset
        dataloader = DataLoader(dataset) # create dataloader

        device = 'cpu'
        new_model = NewNetwork().to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3)

        size = len(dataloader.dataset)
        new_model.train()
        
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction error
            pred = new_model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return None