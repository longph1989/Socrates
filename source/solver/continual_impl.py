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
            # nn.Linear(50, 50),
            # nn.ReLU(),
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



    def __write_bounds(self, prob, lw_coll, up_coll, min_weight, max_weight, num_weights):
        for idx in range(num_weights):
            prob.write('  {} <= w{} <= {}\n'.format(min_weight, idx, max_weight))

        for cnt_imgs in range(len(lw_coll)):
            lw_list = lw_coll[cnt_imgs]
            up_list = up_coll[cnt_imgs]

            for var_idx in range(len(lw_list)):
                lw, up = lw_list[var_idx], up_list[var_idx]
                if lw == up:
                    prob.write('  x{}_{} = {}\n'.format(var_idx, cnt_imgs, lw))
                else:
                    prob.write('  {} <= x{}_{} <= {}\n'.format(lw, var_idx, cnt_imgs, up))


    def __write_binary(self, prob, bins_coll):
        prob.write(' ')
        for cnt_imgs in range(len(bins_coll)):
            for idx in range(bins_coll[cnt_imgs]):
                prob.write(' a{}_{}'.format(idx, cnt_imgs))
        prob.write('\n')


    def __write_objective(self, prob, num_weights, old_weights):
        prob.write('  [ w0 ^ 2')
        for idx in range(1, num_weights):
            prob.write(' + w{} ^ 2 '.format(idx))
        prob.write(' ]')

        for idx in range(num_weights):
            old_weight = old_weights[idx]
            if old_weight > 0.0:
                prob.write(' - {} w{}'.format(2 * old_weight, idx))
            elif old_weight < 0.0:
                prob.write(' + {} w{}'.format(2 * abs(old_weight), idx))
        prob.write('\n')


    def __write_problem(self, model, sample_x0s_with_bd, trigger, mask, target,
            repair_layer, repair_neuron, min_weight, max_weight):
        filename = 'prob.lp'
        prob = open(filename, 'w')

        # fix outgoing weights
        num_weights = model.layers[repair_layer + 2].get_number_neurons()
        old_weights = model.layers[repair_layer + 2].weights[repair_neuron,:].copy()

        prob.write('Minimize\n')
        self.__write_objective(prob, num_weights, old_weights)

        lw_coll, up_coll, bins_coll = [], [], []

        prob.write('Subject To\n')

        cnt_imgs, has_bins = 0, False

        # original input
        for x_0, _, output_x0, _ in sample_x0s_with_bd:
            # compute input up to the next layer
            input_repair = model.apply_to(x_0, repair_layer + 2).reshape(-1)
            y0 = np.argmax(output_x0)

            lw_list, up_list, num_bins = self.__write_constr(prob, model, input_repair, repair_layer, repair_neuron,
                min_weight, max_weight, cnt_imgs, y0)

            if num_bins > 0: has_bins = True
            
            lw_coll.append(lw_list)
            up_coll.append(up_list)
            bins_coll.append(num_bins)
            
            cnt_imgs += 1

        # input with backdoor
        for _, x_bd, output_x0, _ in sample_x0s_with_bd:
            # compute input up to the next layer
            input_repair = model.apply_to(x_bd, repair_layer + 2).reshape(-1)
            y0 = np.argmax(output_x0)

            lw_list, up_list, num_bins = self.__write_constr(prob, model, input_repair, repair_layer, repair_neuron,
                min_weight, max_weight, cnt_imgs, y0)

            if num_bins > 0: has_bins = True
            
            lw_coll.append(lw_list)
            up_coll.append(up_list)
            bins_coll.append(num_bins)
            
            cnt_imgs += 1

        prob.write('Bounds\n')
        self.__write_bounds(prob, lw_coll, up_coll, min_weight, max_weight, num_weights)

        if has_bins:
            prob.write('Binary\n')
            self.__write_binary(prob, bins_coll)

        prob.write('End\n')

        prob.flush()
        prob.close()


    def __write_constr_next_layer(self, prob, repair_neuron, number_of_neurons, lw_prev, up_prev,
            weights, bias, min_weight, max_weight, cnt_imgs, prev_var_idx, curr_var_idx):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = 0.0, 0.0

            for weight_idx in range(len(weights[neuron_idx])):
                if weight_idx == repair_neuron:
                    lw_val, up_val = lw_prev[weight_idx], up_prev[weight_idx]
                    lw += min(min_weight * lw_val, max_weight * lw_val, min_weight * up_val, max_weight * up_val)
                    up += max(min_weight * lw_val, max_weight * lw_val, min_weight * up_val, max_weight * up_val)
                else:
                    weight_val = weights[neuron_idx][weight_idx]
                    if weight_val > 0:
                        lw += weight_val * lw_prev[weight_idx]
                        up += weight_val * up_prev[weight_idx]
                    elif weight_val < 0:
                        lw += weight_val * up_prev[weight_idx]
                        up += weight_val * lw_prev[weight_idx]

            lw, up = lw + bias[neuron_idx], up + bias[neuron_idx]
            assert lw <= up

            lw_layer.append(lw)
            up_layer.append(up)

            # write constraints
            prob.write('  x{}_{}'.format(curr_var_idx + neuron_idx, cnt_imgs))
            coefs = -weights[neuron_idx]
            for coef_idx in range(len(coefs)):
                coef = coefs[coef_idx]
                if coef_idx == repair_neuron:
                    prob.write(' - [ w{} * x{}_{} ]'.format(neuron_idx, prev_var_idx + coef_idx, cnt_imgs))
                else:
                    if coef > 0.0:
                        prob.write(' + {} x{}_{}'.format(coef, prev_var_idx + coef_idx, cnt_imgs))
                    elif coef < 0.0:
                        prob.write(' - {} x{}_{}'.format(abs(coef), prev_var_idx + coef_idx, cnt_imgs))
            prob.write(' = {}\n'.format(bias[neuron_idx]))

        return lw_layer, up_layer


    def __write_constr_other_layers(self, prob, number_of_neurons, lw_prev, up_prev,
            weights, bias, cnt_imgs, prev_var_idx, curr_var_idx):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = 0.0, 0.0

            for weight_idx in range(len(weights[neuron_idx])):
                weight_val = weights[neuron_idx][weight_idx]
                if weight_val > 0:
                    lw += weight_val * lw_prev[weight_idx]
                    up += weight_val * up_prev[weight_idx]
                elif weight_val < 0:
                    lw += weight_val * up_prev[weight_idx]
                    up += weight_val * lw_prev[weight_idx]

            lw, up = lw + bias[neuron_idx], up + bias[neuron_idx]
            assert lw <= up

            lw_layer.append(lw)
            up_layer.append(up)

            # write constraints
            prob.write('  x{}_{}'.format(curr_var_idx + neuron_idx, cnt_imgs))
            coefs = -weights[neuron_idx]
            for coef_idx in range(len(coefs)):
                coef = coefs[coef_idx]
                if coef > 0.0:
                    prob.write(' + {} x{}_{}'.format(coef, prev_var_idx + coef_idx, cnt_imgs))
                elif coef < 0.0:
                    prob.write(' - {} x{}_{}'.format(abs(coef), prev_var_idx + coef_idx, cnt_imgs))
            prob.write(' = {}\n'.format(bias[neuron_idx]))

        return lw_layer, up_layer


    def __write_constr_relu_layers(self, prob, number_of_neurons, lw_prev, up_prev,
            cnt_imgs, prev_var_idx, curr_var_idx, num_bins):
        lw_layer, up_layer = [], []

        for neuron_idx in range(number_of_neurons):
            # compute bounds
            lw, up = lw_prev[neuron_idx], up_prev[neuron_idx]
            assert lw <= up

            lw_layer.append(max(lw, 0.0))
            up_layer.append(max(up, 0.0))

            # write constraints
            if lw < 0.0 and up > 0.0:
                cvar_idx = curr_var_idx + neuron_idx
                pvar_idx = prev_var_idx + neuron_idx

                prob.write('  x{}_{} - x{}_{} + {} a{}_{} <= {}\n'.format(cvar_idx, cnt_imgs, pvar_idx, cnt_imgs, -lw, num_bins, cnt_imgs, -lw))
                prob.write('  x{}_{} - x{}_{} >= 0.0\n'.format(cvar_idx, cnt_imgs, pvar_idx, cnt_imgs))
                prob.write('  x{}_{} - {} a{}_{} <= 0.0\n'.format(cvar_idx, cnt_imgs, up, num_bins, cnt_imgs))
                prob.write('  x{}_{} >= 0.0\n'.format(cvar_idx, cnt_imgs))
                num_bins += 1
            elif lw >= 0.0:
                prob.write('  x{}_{} - x{}_{} = 0.0\n'.format(curr_var_idx + neuron_idx, cnt_imgs, prev_var_idx + neuron_idx, cnt_imgs))

        return lw_layer, up_layer, num_bins


    def __write_constr(self, prob, model, input_repair, repair_layer, repair_neuron,
            min_weight, max_weight, cnt_imgs, y0):
        lw_list, up_list = [], []
        lw_input, up_input = [], []
        num_bins = 0

        for input_val in input_repair:
            lw_input.append(input_val)
            up_input.append(input_val)

        lw_list.append(lw_input)
        up_list.append(up_input)

        curr_var_idx = len(input_repair)
        prev_var_idx = 0

        for layer_idx in range(repair_layer + 2, len(model.layers)):
            layer = model.layers[layer_idx]
            lw_layer, up_layer = [], []

            # fully connected layer
            if layer.is_linear_layer():
                weights = layer.weights.transpose(1, 0) # shape: num_neuron X input
                bias = layer.bias.transpose(1, 0).reshape(-1) # shape: num_neuron
                number_of_neurons = layer.get_number_neurons()

                # next linear layer
                if layer_idx == repair_layer + 2:
                    lw_prev, up_prev = lw_list[-1], up_list[-1]
                    lw_layer, up_layer = self.__write_constr_next_layer(prob, repair_neuron, number_of_neurons, lw_prev, up_prev,
                                            weights, bias, min_weight, max_weight, cnt_imgs, prev_var_idx, curr_var_idx)                 
                # other linear layers
                else:
                    lw_prev, up_prev = lw_list[-1], up_list[-1]
                    lw_layer, up_layer = self.__write_constr_other_layers(prob, number_of_neurons, lw_prev, up_prev,
                                            weights, bias, cnt_imgs, prev_var_idx, curr_var_idx)
            # ReLU
            else:
                lw_prev, up_prev = lw_list[-1], up_list[-1]
                number_of_neurons = len(lw_prev)
                lw_layer, up_layer, num_bins = self.__write_constr_relu_layers(prob, number_of_neurons, lw_prev, up_prev,
                                            cnt_imgs, prev_var_idx, curr_var_idx, num_bins)

            lw_list.append(lw_layer)
            up_list.append(up_layer)

            prev_var_idx = curr_var_idx
            curr_var_idx += len(lw_layer)

        # output constraints
        for output_idx in range(len(lw_list[-1])):
            if output_idx != y0:
                # use 0.001 to guarantee the output condition
                prob.write('  x{}_{} - x{}_{} > 0.001\n'.format(prev_var_idx + y0, cnt_imgs, prev_var_idx + output_idx, cnt_imgs))

        flat_lw_list = [item for sublist in lw_list for item in sublist]
        flat_up_list = [item for sublist in up_list for item in sublist]

        prob.flush()

        return flat_lw_list, flat_up_list, num_bins



    def solve(self, models, assertion, display=None):
        # lower, upper = model.lower, model.upper
        # bound: [(-0.3284,0.6799), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]

        lower = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5])
        upper = np.array([0.6799, 0.5, 0.5, 0.5, 0.5])

        # print("lower = {}".format(lower))
        # print("upper = {}".format(upper))

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
        ys = np.argmin(all_data[:,-5:], axis=1).reshape(-1)

        # print(xs.shape)
        # print(ys.shape)
        # a = np.argmin(ys, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print(dict(zip(unique, counts)))

        # # print(xs)
        # # print("xs.shape = {}".format(xs.shape))
        # # print("ys.shape = {}".format(ys.shape))

        train_x, train_y = xs[:100000], ys[:100000]
        test_x, test_y = xs[100000:], ys[100000:]

        # a = np.argmin(train_y, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print('Train dist = {}'.format(dict(zip(unique, counts))))

        # a = np.argmin(test_y, 1)
        unique, counts = np.unique(test_y, return_counts=True)
        print('Test dist = {}'.format(dict(zip(unique, counts))))

        aux_x, aux_y = [], []

        for i in range(100000):
            if train_y[i] != 0:
                for j in range(9):
                    aux_x.append(train_x[i])
                    aux_y.append(train_y[i])

        train_x = np.concatenate((train_x, np.array(aux_x)), axis=0)
        train_y = np.concatenate((train_y, np.array(aux_y)), axis=0)

        # print(train_x.shape)
        # print(train_y.shape)

        # a = np.argmin(train_y, 1)
        # unique, counts = np.unique(a, return_counts=True)
        # print('Aux train dist = {}'.format(dict(zip(unique, counts))))

        tensor_train_x = torch.Tensor(train_x.copy()) # transform to torch tensor
        tensor_train_y = torch.Tensor(train_y.copy()).type(torch.LongTensor)

        tensor_test_x = torch.Tensor(test_x.copy()) # transform to torch tensor
        tensor_test_y = torch.Tensor(test_y.copy()).type(torch.LongTensor)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y) # create dataset
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True) # create dataloader

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader




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





        device = 'cpu'
        new_model = NewNetwork().to(device)

        optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

        num_of_epochs = 200
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
                self.test(new_model, train_dataloader, loss_fn_arr[i], device)
                self.test(new_model, test_dataloader, loss_fn_arr[i], device)

                # scheduler.step()

        # dataloader_iterator = iter(train_dataloader)
        # aa, _ = next(dataloader_iterator)

        # for i in range(10):
        #     print('\n-------------------------------------\n')
        #     print('i = {}'.format(i))
        #     print(aa[i])
        #     print(new_model(aa[i]))
        #     print(model.apply(aa[i].detach().cpu().numpy()))

        return None