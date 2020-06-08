import autograd.numpy as np

model = None

def set_model(m):
    global model
    model = m

def apply_model(x):
    global model
    return model.apply(x)

def d0(x1, x2):
    return np.sum(x1 != x2)

def d2(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def di(x1, x2):
    return np.max(np.abs(x1 - x2))

def arg_max(x):
    return np.argmax(apply_model(x), axis=1)

def arg_min(x):
    return np.argmin(apply_model(x), axis=1)

def lin_inp(coefs, x):
    res = 0
    for i in range(x.size):
        res = res + coefs[i] * x[i]
    return res

def lin_out(coefs, x):
    out = apply_model(x)
    for i in range(out.size):
        res = res + coefs[i] * out[i]
    return res

def index(i, x):
    return x[i]
