import autograd.numpy as np

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
        self.w = weights.transpose(1, 0)
        self.b = bias.reshape(-1, bias.size)

    def apply(self, x):
        return x @ self.w + self.b


class ReluRNN:
    def __init__(self, weights, bias, h0, len):
        self.w = weights.transpose(1, 0)
        self.b = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

        self.len = len
        self.count = 0

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        self.h_t = relu(x @ self.w + self.b)

        res = self.h_t

        self.count = self.count + 1
        if self.count == self.len:
            self.h_t = self.h_0
            self.count = 0

        return res


class TanhRNN:
    def __init__(self, weights, bias, h0, len):
        self.w = weights.transpose(1, 0)
        self.b = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

        self.len = len
        self.count = 0

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        self.h_t = np.tanh(x @ self.w + self.b)

        res = self.h_t

        self.count = self.count + 1
        if self.count == self.len:
            self.h_t = self.h_0
            self.count = 0

        return res


class LSTM:
    def __init__(self, weights, bias, h0, c0, len):
        self.w = weights.transpose(1, 0)
        self.b = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.c_0 = c0.reshape(-1, c0.size)

        self.h_t = h0.reshape(-1, h0.size)
        self.c_t = c0.reshape(-1, c0.size)

        self.len = len
        self.count = 0

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        gates = x @ self.w + self.b

        i, j, f, o = np.split(gates, 4, axis=1)

        self.c_t = self.c_t * sigmoid(f) + sigmoid(i) * np.tanh(j)
        self.h_t = sigmoid(o) * np.tanh(self.c_t)

        res = self.h_t

        self.count = self.count + 1
        if self.count == self.len:
            self.h_t = self.h_0
            self.c_t = self.c_0
            self.count = 0

        return res


class GRU:
    def __init__(self, gate_weights, candidate_weights,
            gate_bias, candidate_bias, h0, len):
        self.gw = gate_weights.transpose(1, 0)
        self.gb = gate_bias.reshape(-1, gate_bias.size)

        self.cw = candidate_weights.transpose(1, 0)
        self.cb = candidate_bias.reshape(-1, candidate_bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

        self.len = len
        self.count = 0

    def apply(self, x):
        gx = np.concatenate((x, self.h_t), axis=1)

        gates = sigmoid(gx @ self.gw + self.gb)

        r, u = np.split(gates, 2, axis=1)
        r = r * self.h_t

        cx = np.concatenate((x, r), axis=1)
        c = np.tanh(cx @ self.cw + self.cb)

        self.h_t = (1 - u) * c + u * self.h_t

        res = self.h_t

        self.count = self.count + 1
        if self.count == self.len:
            self.h_t = self.h_0
            self.count = 0

        return res


# class GRU:
#     def __init__(self, weights_ir, weights_iz, weights_in,
#                 weights_hr, weights_hz, weights_hn,
#                 bias_ir, bias_iz, bias_in,
#                 bias_hr, bias_hz, bias_hn,
#                 h0):
#         self.w_ir = weights_ir.transpose(1, 0)
#         self.w_iz = weights_iz.transpose(1, 0)
#         self.w_in = weights_in.transpose(1, 0)
#         self.w_hr = weights_hr.transpose(1, 0)
#         self.w_hz = weights_hz.transpose(1, 0)
#         self.w_hn = weights_hn.transpose(1, 0)
#
#         self.b_ir = bias_ir.reshape(-1, bias_ir.size)
#         self.b_iz = bias_iz.reshape(-1, bias_iz.size)
#         self.b_in = bias_in.reshape(-1, bias_in.size)
#         self.b_hr = bias_hr.reshape(-1, bias_hr.size)
#         self.b_hz = bias_hz.reshape(-1, bias_hz.size)
#         self.b_hn = bias_hn.reshape(-1, bias_hn.size)
#
#         self.h_t = h0
#
#     def apply(self, x):
#         r_t = sigmoid(x @ self.w_ir + self.b_ir + self.h_t @ self.w_hr + self.b_hr)
#         z_t = sigmoid(x @ self.w_iz + self.b_iz + self.h_t @ self.w_hz + self.b_hz)
#         n_t = np.tanh(x @ self.w_in + self.b_in + r_t * (self.h_t @ self.w_hn + self.b_hn))
#         self.h_t = (1 - z_t) * n_t + z_t * self.h_t
#
#         return self.h_t


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


class MaxPool1d:
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


class MaxPool2d:
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
        # print(res.shape)

        res = res.reshape(x_c, k_h * k_w, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_h, res_w)

        return res


class MaxPool3d:
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
