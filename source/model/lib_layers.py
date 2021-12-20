import autograd.numpy as np

from solver.refinement_impl import Poly
from utils import *
from poly_utils import *


class Layer:
    def apply(self, x):
        return x

    def reset(self):
        pass

    def copy(self):
        pass


class Function(Layer):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.func = get_func(name, params)

    def copy(self):
        new_layer = Function(self.name, self.params)
        return new_layer

    def apply(self, x):
        return self.func(x)

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        no_neurons = len(x_poly.lw)

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        res.le = np.zeros([no_neurons, no_neurons + 1])
        res.ge = np.zeros([no_neurons, no_neurons + 1])

        res.shape = x_poly.shape
        res.is_activation = True

        if self.name == 'relu':
            for i in range(no_neurons):
                if x_poly.up[i] <= 0:
                    pass
                elif x_poly.lw[i] >= 0:
                    res.le[i,i] = 1
                    res.ge[i,i] = 1

                    res.lw[i] = x_poly.lw[i]
                    res.up[i] = x_poly.up[i]
                else:
                    res.le[i,i] = x_poly.up[i] / (x_poly.up[i] - x_poly.lw[i])
                    res.le[i,-1] = - x_poly.up[i] * x_poly.lw[i] / (x_poly.up[i] - x_poly.lw[i])

                    lam = 0 if x_poly.up[i] <= -x_poly.lw[i] else 1

                    res.ge[i,i] = lam
                    res.lw[i] = 0 # it seems safe to set lw = 0 anyway
                    # res.lw[i] = lam * x_poly.lw[i] # notice: mnist_relu_5_10.tf
                    res.up[i] = x_poly.up[i]

        elif self.name == 'sigmoid':
            res.lw = sigmoid(x_poly.lw)
            res.up = sigmoid(x_poly.up)

            for i in range(no_neurons):
                if x_poly.lw[i] == x_poly.up[i]:
                    res.le[i][-1] = res.lw[i]
                    res.ge[i][-1] = res.lw[i]
                else:
                    if x_poly.lw[i] > 0:
                        lam1 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        if x_poly.up[i] <= 0:
                            lam2 = lam1
                        else:
                            ll = sigmoid(x_poly.lw[i]) * (1 - sigmoid(x_poly.lw[i]))
                            uu = sigmoid(x_poly.up[i]) * (1 - sigmoid(x_poly.up[i]))
                            lam2 = min(ll, uu)
                    else:
                        ll = sigmoid(x_poly.lw[i]) * (1 - sigmoid(x_poly.lw[i]))
                        uu = sigmoid(x_poly.up[i]) * (1 - sigmoid(x_poly.up[i]))
                        lam1 = min(ll, uu)
                        if x_poly.up[i] <= 0:
                            lam2 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        else:
                            lam2 = lam1

                    res.ge[i,i] = lam1
                    res.ge[i,-1] = res.lw[i] - lam1 * x_poly.lw[i]

                    res.le[i,i] = lam2
                    res.le[i,-1] = res.up[i] - lam2 * x_poly.up[i]

        elif self.name == 'tanh':
            res.lw = tanh(x_poly.lw)
            res.up = tanh(x_poly.up)

            for i in range(no_neurons):
                if x_poly.lw[i] == x_poly.up[i]:
                    res.le[i][-1] = res.lw[i]
                    res.ge[i][-1] = res.lw[i]
                else:
                    if x_poly.lw[i] > 0:
                        lam1 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        if x_poly.up[i] <= 0:
                            lam2 = lam1
                        else:
                            ll = 1 - pow(tanh(x_poly.lw[i]), 2)
                            uu = 1 - pow(tanh(x_poly.up[i]), 2)
                            lam2 = min(ll, uu)
                    else:
                        ll = 1 - pow(tanh(x_poly.lw[i]), 2)
                        uu = 1 - pow(tanh(x_poly.up[i]), 2)
                        lam1 = min(ll, uu)
                        if x_poly.up[i] <= 0:
                            lam2 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        else:
                            lam2 = lam1

                    res.ge[i,i] = lam1
                    res.ge[i,-1] = res.lw[i] - lam1 * x_poly.lw[i]

                    res.le[i,i] = lam2
                    res.le[i,-1] = res.up[i] - lam2 * x_poly.up[i]
        
        elif self.name == 'reshape':
            res.lw = x_poly.lw.copy()
            res.up = x_poly.up.copy()

            res.le = np.eye(no_neurons + 1)[:-1]
            res.ge = np.eye(no_neurons + 1)[:-1]

            res.shape = self.params[0]

        return res

    def is_poly_exact(self):
        if self.func == relu or self.func == sigmoid or self.func == tanh:
            return False
        else:
            return True

    def is_activation_layer(self):
        return True

    def is_linear_layer(self):
        return False

    def get_number_neurons(self):
        return None


class Linear(Layer):
    def __init__(self, weights, bias, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)
        self.func = get_func(name, None)

    def copy(self):
        new_layer = Linear(np.zeros((1,1)), np.zeros(1), None)

        new_layer.weights = self.weights.copy()
        new_layer.bias = self.bias.copy()
        new_layer.func = self.func

        return new_layer

    def apply(self, x):
        if self.func == None:
            return x @ self.weights + self.bias
        else:
            return self.func(x @ self.weights + self.bias)


    def get_weight(self):
        return self.weights


    def apply_poly(self, x_poly, lst_poly):
        assert self.func == None, "self.func should be None"

        weights = self.weights.transpose(1, 0)
        bias = self.bias.transpose(1, 0)

        no_neurons = len(bias)

        res = Poly()

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        res.le = np.concatenate([weights, bias], axis=1)
        res.ge = np.concatenate([weights, bias], axis=1)

        res.shape = (1, no_neurons)

        res.back_substitute(lst_poly)

        return res

    def is_poly_exact(self):
        return True

    def is_activaiton_layer(self):
        return False

    def is_linear_layer(self):
        return True

    def get_number_neurons(self):
        return len(self.bias[0])


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

    def get_weight(self):
        return self.weights

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

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        f_n, f_c, f_h, f_w = self.filters.shape
        x_n, x_c, x_h, x_w = x_poly.shape

        x_w += 2 * self.padding
        x_h += 2 * self.padding        

        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1

        len_pad = x_c * x_h * x_w
        len_res = f_n * res_h * res_w

        res.lw = np.zeros(len_res)
        res.up = np.zeros(len_res)

        res.le = np.zeros([len_res, len_pad + 1])
        res.ge = np.zeros([len_res, len_pad + 1])

        res.shape = (1, f_n, res_h, res_w)

        for i in range(f_n):
            base = np.zeros([x_c, x_h, x_w])
            base[:f_c, :f_h, :f_w] = self.filters[i]
            base = np.reshape(base, -1)
            w_idx = f_w

            for j in range(res_h * res_w):
                res.le[i * res_h * res_w + j] = np.append(base, [self.bias[i]])
                res.ge[i * res_h * res_w + j] = np.append(base, [self.bias[i]])

                if w_idx + self.stride <= x_w:
                    base = np.roll(base, self.stride)
                    w_idx += self.stride
                else:
                    base = np.roll(base, self.stride * x_w - w_idx + f_w)
                    w_idx = f_w

        del_idx = []
        if self.padding > 0:
            del_idx = del_idx + list(range(self.padding * (x_w + 1)))
            mx = x_h - self.padding
            for i in range(self.padding + 1, mx):
                tmp = i * x_w
                del_idx = del_idx + list(range(tmp - self.padding, tmp + self.padding))
            del_idx = del_idx + list(range(mx * x_w - self.padding, x_h * x_w))

            tmp = np.array(del_idx)

            for i in range(1, x_c):
                offset = i * x_h * x_w
                del_idx = del_idx + list((tmp + offset).copy())

        res.le = np.delete(res.le, del_idx, 1)
        res.ge = np.delete(res.ge, del_idx, 1)

        res.le = np.ascontiguousarray(res.le)
        res.ge = np.ascontiguousarray(res.ge)

        res.back_substitute(lst_poly)

        return res

    def is_poly_exact(self):
        return True


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

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        k_h, k_w = self.kernel

        lw, up = x_poly.lw.copy(), x_poly.up.copy()

        lw = lw.reshape(x_poly.shape)
        up = up.reshape(x_poly.shape)

        p = self.padding
        lw_pad = np.pad(lw, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        up_pad = np.pad(up, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = lw_pad.shape

        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        len_pad = x_c * x_h * x_w
        len_res = x_c * res_h * res_w

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, self.kernel, (x_h, x_w))

        res_lw, res_up = [], []
        mx_lw_idx_lst, mx_up_idx_lst = [], []

        for c in range(x_c):
            for i in range(res_h * res_w):

                mx_lw_val, mx_lw_idx = -1e9, None
                
                for k in range(k_h * k_w):

                    h, w = h_idx[k, i], w_idx[k, i]
                    val = lw_pad[0, c, h, w]
                    
                    if val > mx_lw_val:
                        mx_lw_val, mx_lw_idx = val, (c, h, w)

                mx_up_val, cnt = -1e9, 0

                for k in range(k_h * k_w):

                    h, w = h_idx[k, i], w_idx[k, i]
                    val = up_pad[0, c, h, w]

                    if val > mx_up_val:
                        mx_up_val = val

                    if mx_lw_idx != (c, h, w) and val > mx_lw_val:
                        cnt += 1

                res_lw.append(mx_lw_val)
                res_up.append(mx_up_val)

                mx_lw_idx_lst.append(mx_lw_idx)
                if cnt > 0: mx_up_idx_lst.append(None)
                else: mx_up_idx_lst.append(mx_lw_idx)


        res.lw = np.array(res_lw)
        res.up = np.array(res_up)

        res.le = np.zeros([len_res, len_pad + 1])
        res.ge = np.zeros([len_res, len_pad + 1])

        res.shape = (1, x_c, res_h, res_w)

        for i in range(len_res):
            c = mx_lw_idx_lst[i][0]
            h = mx_lw_idx_lst[i][1]
            w = mx_lw_idx_lst[i][2]

            idx = c * x_h * x_w + h * x_w + w
            res.ge[i, idx] = 1

            if mx_up_idx_lst[i] is None:
                res.le[i, -1] = res.up[i]
            else:
                res.le[i, idx] = 1 

        del_idx = []
        if self.padding > 0:
            del_idx = del_idx + list(range(self.padding * (x_w + 1)))
            mx = x_h - self.padding
            for i in range(self.padding + 1, mx):
                tmp = i * x_w
                del_idx = del_idx + list(range(tmp - self.padding, tmp + self.padding))
            del_idx = del_idx + list(range(mx * x_h - self.padding, x_h * x_w))

            tmp = np.array(del_idx)

            for i in range(1, x_c):
                offset = i * x_h * x_w
                del_idx = del_idx + list((tmp + offset).copy())

        res.le = np.delete(res.le, del_idx, 1)
        res.ge = np.delete(res.ge, del_idx, 1)

        return res

    def is_poly_exact(self):
        # may be should be False
        return True


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
