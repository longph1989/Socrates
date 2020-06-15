import autograd.numpy as np


model = None
cache = dict()

def set_model(m):
    global model
    model = m

def update_cache(str_x, output):
    cache[str_x] = output

def apply_model(x):
    str_x = str(x)

    if str_x in cache:
        return cache[str_x]

    output = model.apply(x)
    update_cache(str_x, output)
    return output

def d0(x1, x2):
    return np.sum(x1 != x2)

def d2(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def di(x1, x2):
    return np.max(np.abs(x1 - x2))

def arg_max(x):
    return np.argmax(apply_model(x), axis=1)[0]

def arg_min(x):
    return np.argmin(apply_model(x), axis=1)[0]

def lin_inp(x, coefs):
    res = 0
    for i in range(x.size):
        res = res + coefs[i] * x[i]
    return res

def lin_out(x, coefs):
    res = 0
    out = apply_model(x)
    for i in range(out.size):
        res = res + coefs[i] * out[0][i]
    return res

def index(x, i):
    return x[i]
