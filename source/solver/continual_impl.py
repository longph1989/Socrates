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
        self.flatten = nn.Flatten()
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
        # loss = nn.MSELoss()
        loss = nn.CrossEntropyLoss()
        return loss(pred, y)

        # size = y.size()[0]
        # lbls = torch.argmin(y, dim=1)

        # pred_lbls = pred[range(size), lbls]
        # y_lbls = y[range(size), lbls]

        # return torch.mean((pred - y) ** 2) # + 4 * torch.mean((pred_lbls - y_lbls) ** 2)
        # return torch.mean((pred - y) ** 2) + 4 * torch.mean(pred_lbls)


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
                # correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def gen_data(self, model, dims, xs, ys, x, idx, ln, x1, x2):
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

        # lower = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        # upper = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        # print("lower = {}".format(lower))
        # print("upper = {}".format(upper))

        # xs, ys = [], []

        # dims = []
        # for i in range(len(lower)):
        #     dims.append(np.linspace(lower[i], upper[i], num=5))

        # for x1 in range(5):
        #     for x2 in range(9):
        #         model = models[x1][x2]
        #         self.gen_data(model, dims, xs, ys, [], 0, len(lower), x1, x2)
        
        # xs, ys = np.array(xs), np.array(ys)

        # all_data = np.concatenate((xs, ys), axis=1)
        # print(all_data.shape)

        # np.random.shuffle(all_data)
        # print(all_data.shape)

        # xs = all_data[:,:7]
        # ys = np.argmin(all_data[:,-5:], axis=1).reshape(-1)

        # print(xs.shape)
        # print(ys.shape)
        # a = np.argmin(ys, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print(dict(zip(unique, counts)))

        # # print(xs)
        # # print("xs.shape = {}".format(xs.shape))
        # # print("ys.shape = {}".format(ys.shape))

        # train_x, train_y = xs[:100000], ys[:100000]
        # test_x, test_y = xs[100000:], ys[100000:]

        # a = np.argmin(train_y, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print('Train dist = {}'.format(dict(zip(unique, counts))))

        # a = np.argmin(test_y, 1)
        # unique, counts = np.unique(test_y, return_counts=True)
        # print('Test dist = {}'.format(dict(zip(unique, counts))))

        # aux_x, aux_y = [], []

        # for i in range(100000):
        #     if train_y[i] != 0:
        #         for j in range(9):
        #             aux_x.append(train_x[i])
        #             aux_y.append(train_y[i])

        # train_x = np.concatenate((train_x, np.array(aux_x)), axis=0)
        # train_y = np.concatenate((train_y, np.array(aux_y)), axis=0)

        # print(train_x.shape)
        # print(train_y.shape)

        # a = np.argmin(train_y, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print('Aux train dist = {}'.format(dict(zip(unique, counts))))


        ##############################################

        # tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        # tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        # tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        # tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        # train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        # train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        # test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        # test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader




        ##############################################
        # Download training data from open datasets.
        # training_data = datasets.MNIST(
        #     root="data",
        #     train=True,
        #     download=True,
        #     transform=ToTensor(),
        # )

        # # Download test data from open datasets.
        # test_data = datasets.MNIST(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=ToTensor(),
        # )

        # batch_size = 64

        # # Create data loaders.
        # train_dataloader = DataLoader(training_data, batch_size=batch_size)
        # test_dataloader = DataLoader(test_data, batch_size=batch_size)
        ##############################################

        # device = 'cpu'
        # new_model = NewNetwork().to(device)

        # optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

        # num_of_epochs = 200
        # num_of_properties = 0

        # # loss_fn_arr = [self.loss_fn0, self.loss_fn1, self.loss_fn2, self.loss_fn3]
        # loss_fn_arr = [self.loss_fn0]
        
        # for i in range(num_of_properties + 1):
        #     if i == 0:
        #         print('\n------------- Initial -------------\n')
        #     else:
        #         print('\n------------- Property {} -------------\n'.format(i))

        #     for j in range(num_of_epochs):
        #         print('\n------------- Epoch {} -------------\n'.format(j))
        #         self.train(new_model, train_dataloader, loss_fn_arr[i], optimizer, device)
        #         self.test(new_model, train_dataloader, loss_fn_arr[i], device)
        #         self.test(new_model, test_dataloader, loss_fn_arr[i], device)

        #         # scheduler.step()

        # torch.save(new_model.state_dict(), 'new_model.pt')

        saved_model = NewNetwork()
        saved_model.load_state_dict(torch.load('new_model.pt'))
        # self.test(saved_model, train_dataloader, loss_fn_arr[0], device)
        # self.test(saved_model, test_dataloader, loss_fn_arr[0], device)


        lower = np.array([0.6, -0.5, -0.5, 0.45, -0.5, -0.4, -0.4444])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, -0.45, 0.4, 0.4444])

        threshold, alpha, beta, delta = 0.9, 0.05, 0.05, 0.005

        p0 = threshold + delta
        p1 = threshold - delta

        h0 = beta / (1 - alpha)
        h1 = (1 - beta) / alpha

        pr, no = 1, 0

        with torch.no_grad():
            while True:
                x = torch.Tensor(generate_x(7, lower, upper))
                pred = saved_model(x).numpy()

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


        # for name, param in new_model.named_parameters():
        #     if param.requires_grad:
        #         print('name = {}, data = {}\n'.format(name, param.data))

        # dataloader_iterator = iter(train_dataloader)
        # aa, _ = next(dataloader_iterator)

        # for i in range(10):
        #     print('\n-------------------------------------\n')
        #     print('i = {}'.format(i))
        #     print(aa[i])
        #     print(new_model(aa[i]))
        #     print(model.apply(aa[i].detach().cpu().numpy()))

        return None