import autograd.numpy as np
from functools import partial


class Layer:
    def apply(self, x):
        return x

    def reset():
        return


class Function(Layer):
    def __init__(self, name, params):
        self.func = Utils.get_func(name, params)

    def apply(x):
        return self.func(x)


class Linear(Layer):
    def __init__(self, weights, bias, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)
        self.func = Utils.get_func(name, None)

    def apply(self, x):
        if self.func == None:
            return x @ self.weights + self.bias
        else:
            return self.func(x @ self.weights + self.bias)


class ReluRNN(Layer):
    def __init__(self, weights, bias, h0):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        self.h_t = Utils.relu(x @ self.weights + self.bias)

        return self.h_t

    def reset(self):
        self.h_t = self.h_0


class TanhRNN(Layer):
    def __init__(self, weights, bias, h0):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        self.h_t = Utils.tanh(x @ self.weights + self.bias)

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

        self.c_t = self.c_t * Utils.sigmoid(f) + Utils.sigmoid(i) * Utils.tanh(j)
        self.h_t = Utils.sigmoid(o) * Utils.tanh(self.c_t)

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

        gates = Utils.sigmoid(gx @ self.gate_weights + self.gate_bias)

        r, u = np.split(gates, 2, axis=1)
        r = r * self.h_t

        cx = np.concatenate((x, r), axis=1)
        c = Utils.tanh(cx @ self.candidate_weights + self.candidate_bias)

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

        c_idx, l_idx = Utils.index1d(x_c, self.stride, (f_l), (x_l))

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

        c_idx, h_idx, w_idx = Utils.index2d(x_c, self.stride, (f_h, f_w), (x_h, x_w))

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

        c_idx, d_idx, h_idx, w_idx = Utils.index3d(x_c, self.stride, (f_d, f_h, f_w), (x_d, x_h, x_w))

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

        c_idx, l_idx = Utils.index1d(x_c, self.stride, self.kernel, (x_l))

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

        c_idx, h_idx, w_idx = Utils.index2d(x_c, self.stride, self.kernel, (x_h, x_w))

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

        c_idx, d_idx, h_idx, w_idx = Utils.index3d(x_c, self.stride, self.kernel, (x_d, x_h, x_w))

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
        res = Utils.relu(res)
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
        res = Utils.relu(res)
        res = conv2.apply(res)
        res = Utils.relu(res)
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


class Utils:
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
       return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def get_func(name, params):
        if name == None:
            return None
        elif name == 'relu':
            return Utils.relu
        elif name == 'sigmoid':
            return Utils.sigmoid
        elif name == 'tanh':
            return Utils.tanh
        elif name == 'softmax':
            return None
        elif name == 'reshape':
            import numpy as rnp
            return partial(rnp.reshape, newshape=params[0])
        elif name == 'transpose':
            import numpy as rnp
            return partial(rnp.transpose, axes=params[0])
        else:
            raise NameError('Not support yet!')

    def index1d(channel, stride, kshape, xshape):
        k_l = kshape
        x_l = xshape

        c_idx = np.repeat(np.arange(channel), k_l)
        c_idx = c_idx.reshape(-1, 1)

        res_l = int((x_l - k_l) / stride) + 1

        size = channel * k_l

        l_idx = np.tile(stride * np.arange(res_l), size)
        l_idx = l_idx.reshape(size, -1)
        l_off = np.tile(np.arange(k_l), channel)
        l_off = l_off.reshape(size, -1)
        l_idx = l_idx + l_off

        return c_idx, l_idx

    def index2d(channel, stride, kshape, xshape):
        k_h, k_w = kshape
        x_h, x_w = xshape

        c_idx = np.repeat(np.arange(channel), k_h * k_w)
        c_idx = c_idx.reshape(-1, 1)

        res_h = int((x_h - k_h) / stride) + 1
        res_w = int((x_w - k_w) / stride) + 1

        size = channel * k_h * k_w

        h_idx = np.tile(np.repeat(stride * np.arange(res_h), res_w), size)
        h_idx = h_idx.reshape(size, -1)
        h_off = np.tile(np.repeat(np.arange(k_h), k_w), channel)
        h_off = h_off.reshape(size, -1)
        h_idx = h_idx + h_off

        w_idx = np.tile(np.tile(stride * np.arange(res_w), res_h), size)
        w_idx = w_idx.reshape(size, -1)
        w_off = np.tile(np.arange(k_w), channel * k_h)
        w_off = w_off.reshape(size, -1)
        w_idx = w_idx + w_off

        return c_idx, h_idx, w_idx

    def index3d(channel, stride, kshape, xshape):
        k_d, k_h, k_w = kshape
        x_d, x_h, x_w = xshape

        c_idx = np.repeat(np.arange(channel), k_d * k_h * k_w)
        c_idx = c_idx.reshape(-1, 1)

        res_d = int((x_d - k_d) / stride) + 1
        res_h = int((x_h - k_h) / stride) + 1
        res_w = int((x_w - k_w) / stride) + 1

        size = channel * k_d * k_h * k_w

        d_idx = np.tile(np.repeat(stride * np.arange(res_d), res_h * res_w), size)
        d_idx = d_idx.reshape(size, -1)
        d_off = np.tile(np.repeat(np.arange(k_d), k_h * k_w), channel)
        d_off = d_off.reshape(size, -1)
        d_idx = d_idx + d_off

        h_idx = np.tile(np.tile(np.repeat(stride * np.arange(res_h), res_w), res_d), size)
        h_idx = h_idx.reshape(size, -1)
        h_off = np.tile(np.repeat(np.arange(k_h), k_w), channel * k_d)
        h_off = h_off.reshape(size, -1)
        h_idx = h_idx + h_off

        w_idx = np.tile(np.tile(stride * np.arange(res_w), res_d * res_h), size)
        w_idx = w_idx.reshape(size, -1)
        w_off = np.tile(np.arange(k_w), channel * k_d * k_h)
        w_off = w_off.reshape(size, -1)
        w_idx = w_idx + w_off

        return c_idx, d_idx, h_idx, w_idx
