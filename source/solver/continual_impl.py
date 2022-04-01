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
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ContinualImpl():
    def loss_fn0(self, pred, x, y):
        size = y.size()[0]
        lbls = torch.argmin(y, dim=1)

        pred_lbls = pred[range(size), lbls]
        y_lbls = y[range(size), lbls]

        return torch.mean((pred - y) ** 2) + 4 * torch.mean((pred_lbls - y_lbls) ** 2)


    # def loss_fn1(self, pred, x, y):
    #     loss1 = self.loss_fn0(pred, x, y)

    #     if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
    #         -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
    #         -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
    #         0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
    #         -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
    #         if torch.take(pred, torch.tensor([0])) <= 3.9911:
    #             loss2 = 0.0
    #         else:
    #             loss2 = torch.take(pred, torch.tensor([0])) - 3.9911
    #     else:
    #         loss2 = 0.0

    #     return loss1 + loss2


    # def loss_fn2(self, pred, x, y):
    #     loss1 = self.loss_fn0(pred, x, y)

    #     if 0.6 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= 0.6799 and \
    #         -0.5 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.5 and \
    #         -0.5 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
    #         0.45 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
    #         -0.5 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= -0.45:
    #         if torch.argmax(pred) != 0:
    #             loss2 = 0.0
    #         else:
    #             loss2 = torch.take(pred, torch.tensor([0]))
    #     else:
    #         loss2 = 0.0

    #     return loss1 + loss2


    # def loss_fn3(self, pred, x, y):
    #     loss1 = self.loss_fn0(pred, x, y)

    #     if -0.3035 <= torch.take(x, torch.tensor([0])) and torch.take(x, torch.tensor([0])) <= -0.2986 and \
    #         -0.0095 <= torch.take(x, torch.tensor([1])) and torch.take(x, torch.tensor([1])) <= 0.0095 and \
    #         0.4934 <= torch.take(x, torch.tensor([2])) and torch.take(x, torch.tensor([2])) <= 0.5 and \
    #         0.3 <= torch.take(x, torch.tensor([3])) and torch.take(x, torch.tensor([3])) <= 0.5 and \
    #         0.3 <= torch.take(x, torch.tensor([4])) and torch.take(x, torch.tensor([4])) <= 0.5:
    #         if torch.argmin(pred) != 0:
    #             loss2 = 0.0
    #         else:
    #             loss2 = -torch.take(pred, torch.tensor([0]))
    #     else:
    #         loss2 = 0.0

    #     return loss1 + loss2


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
                correct += (pred.argmin(1) == y.argmin(1)).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def gen_data(self, model, dims, xs, ys, x, idx, ln, x1, x2):
        if idx == ln:
            # output based on the model
            new_x = x.copy()
            y = model.apply(np.array(new_x)).reshape(-1)

            # new dims for x
            new_x.append(x1)
            new_x.append(x2)
            new_x = np.array(new_x)

            xs.append(new_x)
            ys.append(y)
        else:
            for val in dims[idx]:
                x.append(val)
                self.gen_data(model, dims, xs, ys, x, idx + 1, ln, x1, x2)
                x.pop()


    # def gen_data2(self, models, xs, ys, lower, upper):
    #     for i in range(100000):
    #         x = generate_x(5, lower, upper)
    #         x1 = random.randint(0, 4)
    #         x2 = random.randint(0, 8)
    #         y = models[x1][x2].apply(x).reshape(-1)

    #         x = x.tolist()
    #         x.append(x1)
    #         x.append(x2)

    #         x = np.array(x)

    #         xs.append(x)
    #         ys.append(y)


    # def gen_data1(self, models, xs, ys, lower, upper):
    #     dims = []

    #     for i in range(len(lower)):
    #         dims.append(np.linspace(lower[i], upper[i], num=2))

    #     for x1 in range(5):
    #         for x2 in range(9):
    #             for d0 in dims[0]:
    #                 for d1 in dims[1]:
    #                     for d2 in dims[2]:
    #                         for d3 in dims[3]:
    #                             for d4 in dims[4]:
    #                                 x = np.array([d0, d1, d2, d3, d4])
    #                                 y = models[x1][x2].apply(x).reshape(-1)

    #                                 x = x.tolist()
    #                                 x.append(x1)
    #                                 x.append(x2)

    #                                 x = np.array(x)

    #                                 xs.append(x)
    #                                 ys.append(y)


    def solve(self, models, assertion, display=None):
        # lower, upper = model.lower, model.upper
        # bound: [(-0.3284,0.6799), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]

        lower = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        print("lower = {}".format(lower))
        print("upper = {}".format(upper))

        # xs, ys = [], []

        # self.gen_data1(models, xs, ys, lower, upper)
        # xs, ys = np.array(xs), np.array(ys)

        # print('first')
        # print(xs[:10])
        # print('=====================')
        # print(ys[:10])
        # print('=====================')

        # print(len(xs))
        # a = np.argmin(ys, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print(dict(zip(unique, counts)))

        xs, ys = [], []

        dims = []
        for i in range(len(lower)):
            dims.append(np.linspace(lower[i], upper[i], num=5))

        for x1 in range(5):
            for x2 in range(9):
                model = models[x1][x2]
                self.gen_data(model, dims, xs, ys, [], 0, len(lower), x1, x2)
        
        xs, ys = np.array(xs), np.array(ys)

        all_data = np.concatenate((xs, ys), axis=1)
        print(all_data.shape)

        np.random.shuffle(all_data)
        print(all_data.shape)

        xs = all_data[:,:7]
        ys = all_data[:,-5:]

        print(xs.shape)
        print(ys.shape)
        a = np.argmin(ys, 1)
        unique, counts = np.unique(a, return_counts=True)
        print(dict(zip(unique, counts)))

        # print(xs)
        # print("xs.shape = {}".format(xs.shape))
        # print("ys.shape = {}".format(ys.shape))

        train_x, train_y = xs[:100000], ys[:100000]
        test_x, test_y = xs[100000:], ys[100000:]

        a = np.argmin(test_y, 1)
        unique, counts = np.unique(a, return_counts=True)
        print(dict(zip(unique, counts)))

        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy())

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy())

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

        device = 'cpu'
        new_model = NewNetwork().to(device)

        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-4, weight_decay=1e-4)

        num_of_epochs = 10
        num_of_properties = 0

        # loss_fn_arr = [self.loss_fn0, self.loss_fn1, self.loss_fn2, self.loss_fn3]
        loss_fn_arr = [self.loss_fn0]
        
        for i in range(num_of_properties + 1):
            if i == 0:
                print('\n------------- Initial -------------\n')
            else:
                print('\n------------- Property {} -------------\n'.format(i))

            for j in range(num_of_epochs):
                print('\n------------- Epoch {} -------------\n'.format(j))
                self.train(new_model, train_dataloader, loss_fn_arr[i], optimizer, device)
                self.test(new_model, test_dataloader, loss_fn_arr[i], device)

        # dataloader_iterator = iter(train_dataloader)
        # aa, _ = next(dataloader_iterator)

        # for i in range(10):
        #     print('\n-------------------------------------\n')
        #     print('i = {}'.format(i))
        #     print(aa[i])
        #     print(new_model(aa[i]))
        #     print(model.apply(aa[i].detach().cpu().numpy()))

        return None