# import numpy as np
#
# def index2d(channel, stride, kshape, xshape):
#     k_h, k_w = kshape
#     x_h, x_w = xshape
#
#     c_idx = np.repeat(np.arange(channel), k_h * k_w)
#     c_idx = c_idx.reshape(-1, 1)
#
#     res_h = int((x_h - k_h) / stride) + 1
#     res_w = int((x_w - k_w) / stride) + 1
#
#     size = channel * k_h * k_w
#
#     h_idx = np.tile(np.repeat(stride * np.arange(res_h), res_w), size)
#     h_idx = h_idx.reshape(size, -1)
#     h_off = np.tile(np.repeat(np.arange(k_h), k_w), channel)
#     h_off = h_off.reshape(size, -1)
#     h_idx = h_idx + h_off
#
#     w_idx = np.tile(np.tile(stride * np.arange(res_w), res_h), size)
#     w_idx = w_idx.reshape(size, -1)
#     w_off = np.tile(np.arange(k_w), channel * k_h)
#     w_off = w_off.reshape(size, -1)
#     w_idx = w_idx + w_off
#
#     return c_idx, h_idx, w_idx
#
#
# c_idx, h_idx, w_idx = index2d(2, 2, (2,2), (4,4))
#
# print('c_idx = {}'.format(c_idx))
# print('h_idx = {}'.format(h_idx))
# print('w_idx = {}'.format(w_idx))
#
# c_idx = c_idx.transpose(1, 0)
# h_idx = h_idx.transpose(1, 0)
# w_idx = w_idx.transpose(1, 0)
#
# print('c_idx = {}'.format(c_idx))
# print('h_idx = {}'.format(h_idx))
# print('w_idx = {}'.format(w_idx))
#
# a = np.random.rand(3,3)
# b = np.pad(a, ((1,1), (1,1)))
# print(a)
# print(b)
#
# c = b[1:4,1:4]
# print(c)

import multiprocessing
from multiprocessing import Pool
import numpy as np


def foo(i):
    return i, i * i


class A:

    def __init__(self):
        self.arr = np.zeros(10)

    def test(self):
        pool = Pool(multiprocessing.cpu_count())
        for i, ii in pool.map(foo, range(10)):
            self.arr[i] = ii

if __name__ == '__main__':
    a = A()
    a.test()
    
    for i in range(10):
        print(a.arr[i])
