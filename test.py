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

# import multiprocessing
# from multiprocessing import Pool
# import numpy as np


# def foo(i):
    # return i, i * i


# class A:

    # def __init__(self):
        # self.arr = np.zeros(10)

    # def test(self):
        # pool = Pool(multiprocessing.cpu_count())
        # for i, ii in pool.map(foo, range(10)):
            # self.arr[i] = ii

# if __name__ == '__main__':
    # a = A()
    # a.test()
    
    # for i in range(10):
        # print(a.arr[i])


# import autograd.numpy as np

# from autograd import grad

# def foo(a):
    # if a[0] + a[1] + a[2] + 100 > 0:
        # return a[0] + a[1] + a[2] + 100
    # else:
        # return 0

# # a = np.random.rand(3)
# a = np.zeros(3)
# g = grad(foo)

# lr = 0.01

# while (foo(a) > 0):
    # print('a = {}'.format(a))
    # print('foo(a) = {}'.format(foo(a)))
    # print('g(a) = {}'.format(g(a)))
    
    # a = a - lr * g(a)
    
# print('a = {}'.format(a))
# print('foo(a) = {}'.format(foo(a)))

import numpy as np
import multiprocessing
import time

def foo(args):
    a, b = args
    print(a)
    b[a] = a
    print(b)

if __name__ == '__main__':
    a = range(10)
    b = np.random.rand(10)
    
    clones = []
    
    for i in a:
        clones.append(b)
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    t1 = time.time()
    pool.map(foo, zip(a, clones))
    # for i in a: print(i)
    
    t2 = time.time()

    print(t2 - t1)
    print(b)