import autograd.numpy as np
import os

from functools import partial, update_wrapper


def read(text):
    if os.path.isfile(text):
        return open(text, 'r').readline()
    else:
        return text

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def get_func(name, params):
    if name == None:
        return None
    elif name.lower() == 'relu':
        return relu
    elif name.lower() == 'sigmoid':
        return sigmoid
    elif name.lower() == 'tanh':
        return tanh
    elif name.lower() == 'softmax':
        # normally softmax only uses in the last layer to return the probabilities
        # between different labels, which is not necessary for our problems for now,
        # will change it later if we have the problems which need the real softmax
        return None
    elif name.lower() == 'reshape':
        import numpy as rnp
        return wrapped_partial(rnp.reshape, newshape=params)
    elif name.lower() == 'transpose':
        import numpy as rnp
        return wrapped_partial(rnp.transpose, axes=params)
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
    
def generate_x(size, lower, upper):
    x = np.random.rand(size)
    x = (upper - lower) * x + lower

    return x
