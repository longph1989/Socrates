import autograd.numpy as np

class Model:
    def __init__(self, shape, layers):
        self.shape = shape
        self.layers = layers

    def apply(x):
        shape_i = [1, self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            for layer in layers:
                output = layer.apply(output)

        for layer in layers:
            layer.reset()

        return output
