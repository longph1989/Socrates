import autograd.numpy as np

from im2col import *

'''
w is in form of [number of neurons] x [number of features]
b is in form of [number of neurons]
x is in form of [number of inputs (1)] x [number of features]

Example:
w = (20, 100); b = (20); x = (1, 100)
w tranpose is (100, 20)
b reshape is (1, 20)
x @ w + b = (1, 20)
'''
class Linear:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def apply(self, x):
        w = np.tranpose(self.weights)
        b = self.bias.reshape(-1, self.bias.size)

        return x @ w + b

'''
This is only 1 layer. The extension to multi layers should be easy.
'''
class ReluRNN:
    def __init__(self, weights_ih, weights_hh, bias_ih, bias_hh, h0):
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.h0 = h0

    def apply(self, x):
        w_ih = np.transpose(self.weights_ih)
        w_hh = np.transpose(self.weights_hh)
        b_ih = self.bias_ih.reshape(-1, self.bias_ih.size)
        b_hh = self.bias_hh.reshape(-1, self.bias_hh.size)
        h_t = self.h0

        for t in range(x[1]):
            x_t = x[0][t].reshape(1, -1)
            h_t = relu(x_t @ w_ih + b_ih + h_t @ w_hh + b_hh)

        return h_t


class TanhRNN:
    def __init__(self, weights_ih, weights_hh, bias_ih, bias_hh, h0):
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.h0 = h0

    def apply(self, x):
        w_ih = np.transpose(self.weights_ih)
        w_hh = np.transpose(self.weights_hh)
        b_ih = self.bias_ih.reshape(-1, self.bias_ih.size)
        b_hh = self.bias_hh.reshape(-1, self.bias_hh.size)
        h_t = self.h0

        for t in range(x[1]):
            x_t = x[0][t].reshape(1, -1)
            h_t = np.tanh(x_t @ w_ih + b_ih + h_t @ w_hh + b_hh)

        return h_t


class LSTM:
    def __init__(self, weights_ii, weights_if, weights_ig, weights_it,
                weights_hi, weights_hf, weights_hg, weights_ht,
                bias_ii, bias_if, bias_ig, bias_it,
                bias_hi, bias_hf, bias_hg, bias_ht,
                h0, c0):
        self.weights_ii = weights_ii
        self.weights_if = weights_if
        self.weights_ig = weights_ig
        self.weights_it = weights_it
        self.weights_hi = weights_hi
        self.weights_hf = weights_hf
        self.weights_hg = weights_hg
        self.weights_ht = weights_ht
        self.bias_ii = bias_ii
        self.bias_if = bias_if
        self.bias_ig = bias_ig
        self.bias_it = bias_it
        self.bias_hi = bias_hi
        self.bias_hf = bias_hf
        self.bias_hg = bias_hg
        self.bias_ht = bias_ht
        self.h0 = h0
        self.c0 = c0

    def apply(self, x):
        w_ii = np.transpose(self.weights_ii)
        w_if = np.transpose(self.weights_if)
        w_ig = np.transpose(self.weights_ig)
        w_it = np.transpose(self.weights_it)
        w_hi = np.transpose(self.weights_hi)
        w_hf = np.transpose(self.weights_hf)
        w_hg = np.transpose(self.weights_hg)
        w_ht = np.transpose(self.weights_ht)
        b_ii = self.bias_ii.reshape(-1, self.bias_ii.size)
        b_if = self.bias_if.reshape(-1, self.bias_if.size)
        b_ig = self.bias_ig.reshape(-1, self.bias_ig.size)
        b_it = self.bias_it.reshape(-1, self.bias_it.size)
        b_hi = self.bias_hi.reshape(-1, self.bias_hi.size)
        b_hf = self.bias_hf.reshape(-1, self.bias_hf.size)
        b_hg = self.bias_hg.reshape(-1, self.bias_hg.size)
        b_ht = self.bias_ht.reshape(-1, self.bias_ht.size)
        h_t = self.h0
        c_t = self.c0

        for t in range(x[1]):
            x_t = x[0][t].reshape(1, -1)
            i_t = sigmoid(x_t @ w_ii + b_ii + h_t @ w_hi + b_hi)
            f_t = sigmoid(x_t @ w_if + b_if + h_t @ w_hf + b_hf)
            g_t = np.tanh(x_t @ w_ig + b_ig + h_t @ w_hg + b_hg)
            o_t = sigmoid(x_t @ w_io + b_io + h_t @ w_ho + b_ho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)

        return h_t


class GRU:
    def __init__(self, weights_ir, weights_iz, weights_in,
                weights_hr, weights_hz, weights_hn,
                bias_ir, bias_iz, bias_in,
                bias_hr, bias_hz, bias_hn,
                h0):
        self.weights_ir = weights_ir
        self.weights_iz = weights_iz
        self.weights_in = weights_in
        self.weights_hr = weights_hr
        self.weights_hz = weights_hz
        self.weights_hn = weights_hn
        self.bias_ir = bias_ir
        self.bias_iz = bias_iz
        self.bias_in = bias_in
        self.bias_hr = bias_hr
        self.bias_hz = bias_hz
        self.bias_hn = bias_hn
        self.h0 = h0

    def apply(self, x):
        w_ir = np.transpose(self.weights_ir)
        w_iz = np.transpose(self.weights_iz)
        w_in = np.transpose(self.weights_in)
        w_hr = np.transpose(self.weights_hr)
        w_hz = np.transpose(self.weights_hz)
        w_hn = np.transpose(self.weights_hn)
        b_ir = self.bias_ir.reshape(-1, self.bias_ir.size)
        b_iz = self.bias_iz.reshape(-1, self.bias_iz.size)
        b_in = self.bias_in.reshape(-1, self.bias_in.size)
        b_hr = self.bias_hr.reshape(-1, self.bias_hr.size)
        b_hz = self.bias_hz.reshape(-1, self.bias_hz.size)
        b_hn = self.bias_hn.reshape(-1, self.bias_hn.size)
        h_t = self.h0

        for t in range(x[1]):
            x_t = x[0][t].reshape(1, -1)
            r_t = sigmoid(x_t @ w_ir + b_ir + h_t @ w_hr + b_hr)
            z_t = sigmoid(x_t @ w_iz + b_iz + h_t @ w_hz + b_hz)
            n_t = np.tanh(x_t @ w_in + b_in + r_t * (h_t @ w_hn + b_hn))
            h_t = (1 - z_t) * n_t + z_t * h_t

        return h_t


class Conv1d:
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_l = self.filters.shape # 2, 3, 4
        f = self.filters.reshape(f_n, -1)  # 2, 12

        b = self.bias.reshape(f_n, -1)  # 2, 1

        x_pad = np.pad(x, (0, 0, self.padding), mode='constant')
        x_n, x_c, x_l = x_pad.shape  # 1, 3, 10

        # c_idx = np.repeat(np.arange(f_c), f_l) # [0 0 0 0 1 1 1 1 2 2 2 2] = 1, 12
        # c_idx = c_idx.reshape(-1, 1) # 12, 1
        #
        # res_l = int((x_l - f_l) / self.stride) + 1
        #
        # size = f_c * f_l
        #
        # l_idx = np.tile(self.stride * np.arange(res_l), size) # (0 .. 9 0 .. 9 0 .. 9) = 1, 120
        # l_idx = l_idx.reshape(size, -1) # 12, 10
        # l_off = np.tile(np.arange(f_l), f_c) # (0 1 2 3 0 1 2 3 0 1 2 3) = 1, 12
        # l_off = l_off.reshape(size, -1) # 12, 1
        # l_idx = l_idx + l_off # 12, 10

        c_idx, l_idx = index1d(x_c, self.stride, (f_l), (x_l))

        res = x_pad[:, c_idx, l_idx] # 1, 12, 10
        res = res.transpose(1, 0).reshape(size, -1) # 12, 10

        res = f @ res + b # 2, 10
        res = res.reshape(1, f_n, res_l) # 1, 2, 10

        return res


class Conv2d:
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        x_pad = np.pad(x, (0, 0, self.padding, self.padding), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        # c_idx = np.repeat(np.arange(f_c), f_h * f_w)
        # c_idx = c_idx.reshape(-1, 1)
        #
        # res_h = int((x_h - f_h) / self.stride) + 1
        # res_w = int((x_w - f_w) / self.stride) + 1
        #
        # size = f_c * f_h * f_w
        #
        # h_idx = np.tile(np.repeat(step.stride * np.arange(res_h), res_w), size)
        # h_idx = h_idx.reshape(size, -1)
        # h_off = np.tile(np.repeat(np.arange(f_h), f_w), f_c)
        # h_off = h_off.reshape(size, -1)
        # h_idx = h_idx + h_off
        #
        # w_idx = np.tile(np.tile(step.stride * np.arange(res_w), res_h), size)
        # w_idx = w_idx.reshape(size, -1)
        # w_off = np.tile(np.arange(f_w), f_c * f_h)
        # w_off = w_off.reshape(size, -1)
        # w_idx = w_idx + w_off

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, (f_h, f_w), (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]
        res = res.transpose(1, 2, 0).reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_h, res_w)

        return res


class Conv3d:
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_d, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        x_pad = np.pad(x, (0, 0, self.padding, self.padding, self.padding), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        # c_idx = np.repeat(np.arange(f_c), f_d * f_h * f_w)
        # c_idx = c_idx.reshape(-1, 1)
        #
        # res_d = int((x_d - f_d) / self.stride) + 1
        # res_h = int((x_h - f_h) / self.stride) + 1
        # res_w = int((x_w - f_w) / self.stride) + 1
        #
        # size = f_c * f_d * f_h * f_w
        #
        # d_idx = np.tile(np.repeat(step.stride * np.arange(res_d), res_h * res_w), size)
        # d_idx = d_idx.reshape(size, -1)
        # d_off = np.tile(np.repeat(np.arange(f_d), f_h * f_w), f_c)
        # d_off = d_off.reshape(size, -1)
        # d_idx = d_idx + d_off
        #
        # h_idx = np.tile(np.tile(np.repeat(step.stride * np.arange(res_h), res_w), res_d), size)
        # h_idx = h_idx.reshape(size, -1)
        # h_off = np.tile(np.repeat(np.arange(f_h), f_w), f_c * f_d)
        # h_off = h_off.reshape(size, -1)
        # h_idx = h_idx + h_off
        #
        # w_idx = np.tile(np.tile(step.stride * np.arange(res_w), res_d * res_h), size)
        # w_idx = w_idx.reshape(size, -1)
        # w_off = np.tile(np.arange(f_w), f_c * f_d * f_h)
        # w_off = w_off.reshape(size, -1)
        # w_idx = w_idx + w_off

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, (f_d, f_h, f_w), (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.transpose(1, 2, 3, 0).reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_d, res_h, res_w)

        return res


class MaxPool1d:
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        # k_l = self.kernel

        x_pad = np.pad(x, (0, 0, self.padding), mode='constant')
        x_n, x_c, x_l = x_pad.shape

        # c_idx = np.repeat(np.arange(x_c), k_l)
        # c_idx = c_idx.reshape(-1, 1)
        #
        # res_l = int((x_l - k_l) / self.stride) + 1
        #
        # size = x_c * k_l
        #
        # l_idx = np.tile(self.stride * np.arange(res_l), size)
        # l_idx = l_idx.reshape(size, -1)
        # l_off = np.tile(np.arange(k_l), x_c)
        # l_off = l_off.reshape(size, -1)
        # l_idx = l_idx + l_off

        c_idx, l_idx = index1d(x_c, self.stride, self.kernel, (x_l))

        res = x_pad[:, c_idx, l_idx]
        res = res.transpose(1, 0).reshape(size, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_l)

        return res


class MaxPool2d:
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        # k_h, k_w = self.kernel

        x_pad = np.pad(x, (0, 0, self.padding, self.padding), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        # c_idx = np.repeat(np.arange(x_c), k_h * k_w)
        # c_idx = c_idx.reshape(-1, 1)
        #
        # res_h = int((x_h - k_h) / self.stride) + 1
        # res_w = int((x_w - k_w) / self.stride) + 1
        #
        # size = x_c * k_h * k_w
        #
        # h_idx = np.tile(np.repeat(step.stride * np.arange(res_h), res_w), size)
        # h_idx = h_idx.reshape(size, -1)
        # h_off = np.tile(np.repeat(np.arange(k_h), k_w), x_c)
        # h_off = h_off.reshape(size, -1)
        # h_idx = h_idx + h_off
        #
        # w_idx = np.tile(np.tile(step.stride * np.arange(res_w), res_h), size)
        # w_idx = w_idx.reshape(size, -1)
        # w_off = np.tile(np.arange(k_w), x_c * k_h)
        # w_off = w_off.reshape(size, -1)
        # w_idx = w_idx + w_off

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, self.kernel, (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]
        res = res.transpose(1, 2, 0).reshape(size, -1)

        res = np.max(res, axis=2)
        res = res.reshape(1, x_c, res_h, res_w)

        return res


class MaxPool3d:
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        # k_d, k_h, k_w = self.kernel

        x_pad = np.pad(x, (0, 0, self.padding, self.padding, self.padding), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        # c_idx = np.repeat(np.arange(x_c), k_d * k_h * k_w)
        # c_idx = c_idx.reshape(-1, 1)
        #
        # res_d = int((x_d - k_d) / self.stride) + 1
        # res_h = int((x_h - k_h) / self.stride) + 1
        # res_w = int((x_w - k_w) / self.stride) + 1
        #
        # size = x_c * k_d * k_h * k_w
        #
        # d_idx = np.tile(np.repeat(step.stride * np.arange(res_d), res_h * res_w), size)
        # d_idx = d_idx.reshape(size, -1)
        # d_off = np.tile(np.repeat(np.arange(k_d), k_h * k_w), x_c)
        # d_off = d_off.reshape(size, -1)
        # d_idx = d_idx + d_off
        #
        # h_idx = np.tile(np.tile(np.repeat(step.stride * np.arange(res_h), res_w), res_d), size)
        # h_idx = h_idx.reshape(size, -1)
        # h_off = np.tile(np.repeat(np.arange(k_h), k_w), x_c * k_d)
        # h_off = h_off.reshape(size, -1)
        # h_idx = h_idx + h_off
        #
        # w_idx = np.tile(np.tile(step.stride * np.arange(res_w), res_d * res_h), size)
        # w_idx = w_idx.reshape(size, -1)
        # w_off = np.tile(np.arange(k_w), x_c * k_d * k_h)
        # w_off = w_off.reshape(size, -1)
        # w_idx = w_idx + w_off

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, self.kernel, (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.transpose(1, 2, 3, 0).reshape(size, -1)

        res = np.max(res, axis=3)
        res = res.reshape(1, x_c, res_d, res_h, res_w)

        return res


def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


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

    res_h = int((x_h - k_h) / self.stride) + 1
    res_w = int((x_w - k_w) / self.stride) + 1

    size = channel * k_h * k_w

    h_idx = np.tile(np.repeat(step.stride * np.arange(res_h), res_w), size)
    h_idx = h_idx.reshape(size, -1)
    h_off = np.tile(np.repeat(np.arange(k_h), k_w), channel)
    h_off = h_off.reshape(size, -1)
    h_idx = h_idx + h_off

    w_idx = np.tile(np.tile(step.stride * np.arange(res_w), res_h), size)
    w_idx = w_idx.reshape(size, -1)
    w_off = np.tile(np.arange(k_w), channel * k_h)
    w_off = w_off.reshape(size, -1)
    w_idx = w_idx + w_off

    return h_idx, w_idx


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
