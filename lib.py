import autograd.numpy as np


def linear(x, w, b):
    return np.matmul(x, w) + b
