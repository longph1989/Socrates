import autograd.numpy as np
from utils import *


class Layer:
    def apply(self, x):
        return x

    def reset(self):
        pass


class Function(Layer):
    def __init__(self, name, params):
        self.func = get_func(name, params)

    def apply(self, x):
        return self.func(x)

    def apply_poly(self, x, x0_poly):
        # x = (50,785)
        res = Poly()

        no_features = len(x0_poly.lt[0])
        no_neurons = len(x.lw)

        res.lt = np.zeros([no_neurons, no_features])
        res.gt = np.zeros([no_neurons, no_features])

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        if self.func == relu:
            for i in range(no_neurons):
                if x.up[i] <= 0:
                    pass
                elif x.lw[i] >= 0:
                    res.lt[i] = x.lt[i]
                    res.gt[i] = x.gt[i]

                    res.lw[i] = x.lw[i]
                    res.up[i] = x.up[i]
                else:
                    # choose lambda = 0
                    res.lt[i] = x.up[i] * (x.lt[i] - x.lw[i]) / (x.up[i] - x.lw[i])
                    res.up[i] = x.up[i]

        elif self.func == sigmoid:
            res.lw = sigmoid(x.lw)
            res.up = sigmoid(x.up)

            for i in range(no_neurons):
                if res.lw[i] == res.up[i]:
                    res.lt[i][no_features - 1] = res.lw[i]
                    res.gt[i][no_features - 1] = res.lw[i]
                else:
                    if res.lw[i] > 0:
                        lam = (res.up[i] - res.lw[i]) / (x.up[i] - x.lw[i])
                        res.gt[i] = res.lw[i] + lam * (x.gt[i] - x.lw[i])
                    else:
                        ll = sigmoid(x.lw[i]) * (1 - sigmoid(x.lw[i]))
                        uu = sigmoid(x.up[i]) * (1 - sigmoid(x.up[i]))
                        lam = min(ll, uu)
                        res.gt[i] = res.lw[i] + lam * (x.gt[i] - x.lw[i])

                    if res.up[i] <= 0:
                        lam = (res.up[i] - res.lw[i]) / (x.up[i] - x.lw[i])
                        res.lt[i] = res.up[i] + lam * (x.lt[i] - x.up[i])
                    else:
                        ll = sigmoid(x.lw[i]) * (1 - sigmoid(x.lw[i]))
                        uu = sigmoid(x.up[i]) * (1 - sigmoid(x.up[i]))
                        lam = min(ll, uu)
                        res.lt[i] = res.up[i] + lam * (x.lt[i] - x.up[i])

        elif self.func == tanh:
            res.lw = tanh(x.lw)
            res.up = tanh(x.up)

            for i in range(no_neurons):
                if res.lw[i] == res.up[i]:
                    res.lt[i][no_features - 1] = res.lw[i]
                    res.gt[i][no_features - 1] = res.lw[i]
                else:
                    if res.lw[i] > 0:
                        lam = (res.up[i] - res.lw[i]) / (x.up[i] - x.lw[i])
                        res.gt[i] = res.lw[i] + lam * (x.gt[i] - x.lw[i])
                    else:
                        ll = 1 - pow(tanh(x.lw[i]), 2)
                        uu = 1 - pow(tanh(x.up[i]), 2)
                        lam = min(ll, uu)
                        res.gt[i] = res.lw[i] + lam * (x.gt[i] - x.lw[i])

                    if res.up[i] <= 0:
                        lam = (res.up[i] - res.lw[i]) / (x.up[i] - x.lw[i])
                        res.lt[i] = res.up[i] + lam * (x.lt[i] - x.up[i])
                    else:
                        ll = 1 - pow(tanh(x.lw[i]), 2)
                        uu = 1 - pow(tanh(x.up[i]), 2)
                        lam = min(ll, uu)
                        res.lt[i] = res.up[i] + lam * (x.lt[i] - x.up[i])


class Linear(Layer):
    def __init__(self, weights, bias, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)
        self.func = get_func(name, None)

    def apply(self, x):
        if self.func == None:
            return x @ self.weights + self.bias
        else:
            return self.func(x @ self.weights + self.bias)

    def apply_poly(self, x, x0_poly):
        # x = (784,785)
        # weights = (784,50)
        # bias = (1,50)
        # res = (50,785)

        res = Poly()

        no_features = len(x0_poly.lw)
        no_neurons = len(self.weights[0])

        res.lt = np.zeros([no_neurons, no_features + 1])
        res.gt = np.zeros([no_neurons, no_features + 1])

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        for i in range(no_neurons): # 0 to 50
            for j in range(no_features): # 0 to 784
                if self.weights[j,i] > 0:
                    res.lt[i] = res.lt[i] + x.lt[j] * self.weights[j,i]
                else:
                    res.lt[i] = res.lt[i] + x.gt[j] * self.weights[j,i]

                if self.weights[j,i] > 0:
                    res.gt[i] = res.gt[i] + x.gt[j] * self.weights[j,i]
                else:
                    res.gt[i] = res.gt[i] + x.lt[j] * self.weights[j,i]

            res.lt[i,-1] = res.lt[i,-1] + self.bias[0,i]
            res.gt[i,-1] = res.gt[i,-1] + self.bias[0,i]

        for i in range(no_neurons):
            for j in range(no_features):
                if res.gt[i,j] > 0:
                    res.lw[i] = res.lw[i] + res.gt[i,j] * x0_poly.lw[j]
                else:
                    res.lw[i] = res.lw[i] + res.gt[i,j] * x0_poly.up[j]

                if res.lt[i,j] > 0:
                    res.up[i] = res.up[i] + res.lt[i,j] * x0_poly.up[j]
                else:
                    res.up[i] = res.up[i] + res.lt[i,j] * x0_poly.lw[j]

            res.lw[i] = res.lw[i] + res.gt[i,-1]
            res.up[i] = res.up[i] + res.lt[i,-1]

        if self.func != None:
            if self.func == relu:
                func = Function('relu', None)
            elif self.func == sigmoid:
                func = Function('sigmoid', None)
            elif self.func == tanh:
                func = Function('tanh', None)

            res = func.apply_poly(res, x0_poly)

        return res


class BasicRNN(Layer):
    def __init__(self, weights, bias, h0, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

        self.func = get_func(name, None)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        if self.func == None:
            self.h_t = x @ self.weights + self.bias
        else:
            self.h_t = self.func(x @ self.weights + self.bias)

        return self.h_t

    def reset(self):
        self.h_t = self.h_0


class LSTM(Layer):
    def __init__(self, weights, bias, h0, c0):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.c_0 = c0.reshape(-1, c0.size)

        self.h_t = h0.reshape(-1, h0.size)
        self.c_t = c0.reshape(-1, c0.size)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        gates = x @ self.weights + self.bias

        i, j, f, o = np.split(gates, 4, axis=1)

        self.c_t = self.c_t * sigmoid(f) + sigmoid(i) * tanh(j)
        self.h_t = sigmoid(o) * tanh(self.c_t)

        return self.h_t

    def reset(self):
        self.h_t = self.h_0
        self.c_t = self.c_0


class GRU(Layer):
    def __init__(self, gate_weights, candidate_weights,
            gate_bias, candidate_bias, h0):
        self.gate_weights = gate_weights.transpose(1, 0)
        self.gate_bias = gate_bias.reshape(-1, gate_bias.size)

        self.candidate_weights = candidate_weights.transpose(1, 0)
        self.candidate_bias = candidate_bias.reshape(-1, candidate_bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

    def apply(self, x):
        gx = np.concatenate((x, self.h_t), axis=1)

        gates = sigmoid(gx @ self.gate_weights + self.gate_bias)

        r, u = np.split(gates, 2, axis=1)
        r = r * self.h_t

        cx = np.concatenate((x, r), axis=1)
        c = tanh(cx @ self.candidate_weights + self.candidate_bias)

        self.h_t = (1 - u) * c + u * self.h_t

        return self.h_t

    def reset(self):
        self.h_t = self.h_0


class Conv1d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_l = self.filters.shape # 2, 3, 4
        f = self.filters.reshape(f_n, -1)  # 2, 12

        b = self.bias.reshape(f_n, -1)  # 2, 1

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape  # 1, 3, 10

        res_l = int((x_l - f_l) / self.stride) + 1
        size = f_c * f_l

        c_idx, l_idx = index1d(x_c, self.stride, (f_l), (x_l))

        res = x_pad[:, c_idx, l_idx] # 1, 12, 10
        res = res.reshape(size, -1) # 12, 10

        res = f @ res + b # 2, 10
        res = res.reshape(1, f_n, res_l) # 1, 2, 10

        return res


class Conv2d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1
        size = f_c * f_h * f_w

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, (f_h, f_w), (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]
        res = res.reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_h, res_w)

        return res


class Conv3d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_d, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        res_d = int((x_d - f_d) / self.stride) + 1
        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1
        size = f_c * f_d * f_h * f_w

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, (f_d, f_h, f_w), (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_d, res_h, res_w)

        return res


class MaxPool1d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_l = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape

        res_l = int((x_l - k_l) / self.stride) + 1

        c_idx, l_idx = index1d(x_c, self.stride, self.kernel, (x_l))

        res = x_pad[:, c_idx, l_idx]
        res = res.reshape(x_c, k_l, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_l)

        return res


class MaxPool2d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_h, k_w = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, self.kernel, (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]

        res = res.reshape(x_c, k_h * k_w, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_h, res_w)

        return res


class MaxPool3d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_d, k_h, k_w = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        res_d = int((x_d - k_d) / self.stride) + 1
        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, self.kernel, (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.reshape(x_c, k_d * k_h * k_w, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_d, res_h, res_w)

        return res


class ResNet2l(Layer):
    def __init__(self, filters1, bias1, stride1, padding1,
        filters2, bias2, stride2, padding2,
        filtersX=None, biasX=None, strideX=None, paddingX=None):

        self.filters1 = filters1
        self.bias1 = bias1
        self.stride1 = stride1
        self.padding1 = padding1

        self.filters2 = filters2
        self.bias2 = bias2
        self.stride2 = stride2
        self.padding2 = padding2

        self.filtersX = filtersX
        self.biasX = biasX
        self.strideX = strideX
        self.paddingX = paddingX

    def apply(self, x):
        if len(self.filters1.shape) == 3:
            conv1 = Conv1d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv1d(self.filter2, self.bias2, self.stride2, self.padding2)
        elif len(self.filters1.shape) == 4:
            conv1 = Conv2d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv2d(self.filter2, self.bias2, self.stride2, self.padding2)
        elif len(self.filters1.shape) == 5:
            conv1 = Conv3d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv3d(self.filter2, self.bias2, self.stride2, self.padding2)

        res = conv1.apply(x)
        res = relu(res)
        res = conv2.apply(res)

        if self.filterX:
            if len(self.filtersX.shape) == 3:
                convX = Conv1d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 4:
                convX = Conv2d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 5:
                convX = Conv3d(self.filterX, self.biasX, self.strideX, self.paddingX)

            x = convX.apply(x)

        res = res + x

        return res


class ResNet3l(Layer):
    def __init__(self, filters1, bias1, stride1, padding1,
        filters2, bias2, stride2, padding2,
        filters3, bias3, stride3, paddind3,
        filtersX=None, biasX=None, strideX=None, paddingX=None):

        self.filters1 = filters1
        self.bias1 = bias1
        self.stride1 = stride1
        self.padding1 = padding1

        self.filters2 = filters2
        self.bias2 = bias2
        self.stride2 = stride2
        self.padding2 = padding2

        self.filters3 = filters3
        self.bias3 = bias3
        self.stride3 = stride3
        self.padding3 = padding3

        self.filtersX = filtersX
        self.biasX = biasX
        self.strideX = strideX
        self.paddingX = paddingX

    def apply(self, x):
        if len(self.filters1.shape) == 3:
            conv1 = Conv1d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv1d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv1d(self.filter3, self.bias3, self.stride3, self.padding3)
        elif len(self.filters1.shape) == 4:
            conv1 = Conv2d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv2d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv2d(self.filter3, self.bias3, self.stride3, self.padding3)
        elif len(self.filters1.shape) == 5:
            conv1 = Conv3d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv3d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv3d(self.filter3, self.bias3, self.stride3, self.padding3)

        res = conv1.apply(x)
        res = relu(res)
        res = conv2.apply(res)
        res = relu(res)
        res = conv3.apply(res)

        if self.filterX:
            if len(self.filtersX.shape) == 3:
                convX = Conv1d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 4:
                convX = Conv2d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 5:
                convX = Conv3d(self.filterX, self.biasX, self.strideX, self.paddingX)

            x = convX.apply(x)

        res = res + x

        return res
