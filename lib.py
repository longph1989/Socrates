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
        self.w = np.tranpose(weights)
        self.b = bias.reshape(-1, bias.size)

    def apply(self, x):
        return x @ self.w + self.b


class ReluRNN:
    def __init__(self, weights_ih, weights_hh, bias_ih, bias_hh, h0):
        self.w_ih = np.transpose(weights_ih)
        self.w_hh = np.transpose(weights_hh)
        self.b_ih = bias_ih.reshape(-1, bias_ih.size)
        self.b_hh = bias_hh.reshape(-1, bias_hh.size)
        self.h_t = h0

    def apply(self, x):
        self.h_t = relu(x @ self.w_ih + self.b_ih + self.h_t @ self.w_hh + self.b_hh)

        return self.h_t


class TanhRNN:
    def __init__(self, weights_ih, weights_hh, bias_ih, bias_hh, h0):
        self.w_ih = np.transpose(weights_ih)
        self.w_hh = np.transpose(weights_hh)
        self.b_ih = bias_ih.reshape(-1, bias_ih.size)
        self.b_hh = bias_hh.reshape(-1, bias_hh.size)
        self.h_t = h0

    def apply(self, x):
        self.h_t = tanh(x @ self.w_ih + self.b_ih + self.h_t @ self.w_hh + self.b_hh)

        return self.h_t


class LSTM:
    def __init__(self, weights_ii, weights_if, weights_ig, weights_it,
                weights_hi, weights_hf, weights_hg, weights_ht,
                bias_ii, bias_if, bias_ig, bias_it,
                bias_hi, bias_hf, bias_hg, bias_ht,
                h0, c0):
        self.w_ii = np.transpose(weights_ii)
        self.w_if = np.transpose(weights_if)
        self.w_ig = np.transpose(weights_ig)
        self.w_it = np.transpose(weights_it)
        self.w_hi = np.transpose(weights_hi)
        self.w_hf = np.transpose(weights_hf)
        self.w_hg = np.transpose(weights_hg)
        self.w_ht = np.transpose(weights_ht)

        self.b_ii = bias_ii.reshape(-1, bias_ii.size)
        self.b_if = bias_if.reshape(-1, bias_if.size)
        self.b_ig = bias_ig.reshape(-1, bias_ig.size)
        self.b_it = bias_it.reshape(-1, bias_it.size)
        self.b_hi = bias_hi.reshape(-1, bias_hi.size)
        self.b_hf = bias_hf.reshape(-1, bias_hf.size)
        self.b_hg = bias_hg.reshape(-1, bias_hg.size)
        self.b_ht = bias_ht.reshape(-1, bias_ht.size)

        self.h_t = h0
        self.c_t = c0

    def apply(self, x):
        i_t = sigmoid(x @ self.w_ii + self.b_ii + self.h_t @ self.w_hi + self.b_hi)
        f_t = sigmoid(x @ self.w_if + self.b_if + self.h_t @ self.w_hf + self.b_hf)
        g_t = np.tanh(x @ self.w_ig + self.b_ig + self.h_t @ self.w_hg + self.b_hg)
        o_t = sigmoid(x @ self.w_io + self.b_io + self.h_t @ self.w_ho + self.b_ho)
        self.c_t = f_t * self.c_t + i_t * g_t
        self.h_t = o_t * np.tanh(self.c_t)

        return self.h_t


class GRU:
    def __init__(self, weights_ir, weights_iz, weights_in,
                weights_hr, weights_hz, weights_hn,
                bias_ir, bias_iz, bias_in,
                bias_hr, bias_hz, bias_hn,
                h0):
        self.w_ir = np.transpose(weights_ir)
        self.w_iz = np.transpose(weights_iz)
        self.w_in = np.transpose(weights_in)
        self.w_hr = np.transpose(weights_hr)
        self.w_hz = np.transpose(weights_hz)
        self.w_hn = np.transpose(weights_hn)

        self.b_ir = bias_ir.reshape(-1, bias_ir.size)
        self.b_iz = bias_iz.reshape(-1, bias_iz.size)
        self.b_in = bias_in.reshape(-1, bias_in.size)
        self.b_hr = bias_hr.reshape(-1, bias_hr.size)
        self.b_hz = bias_hz.reshape(-1, bias_hz.size)
        self.b_hn = bias_hn.reshape(-1, bias_hn.size)

        self.h_t = h0

    def apply(self, x):
        r_t = sigmoid(x @ self.w_ir + self.b_ir + self.h_t @ self.w_hr + self.b_hr)
        z_t = sigmoid(x @ self.w_iz + self.b_iz + self.h_t @ self.w_hz + self.b_hz)
        n_t = np.tanh(x @ self.w_in + self.b_in + r_t * (self.h_t @ self.w_hn + self.b_hn))
        self.h_t = (1 - z_t) * n_t + z_t * self.h_t

        return self.h_t


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
