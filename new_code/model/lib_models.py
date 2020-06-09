import autograd.numpy as np
import torch

class Model:
    def __init__(self, shape, lower, upper, layers, path):
        self.shape = shape
        self.lower = lower
        self.upper = upper
        self.layers = layers

        if layers == None and path != None:
            self.ptmodel = torch.load(path)

    def __apply_ptmodel(x):
        x = torch.from_numpy(x).view(self.shape)

        with torch.no_grad():
            output = self.ptmodel(x)

        output = output.numpy()

        return output

    def apply(x):
        if self.layers == None:
            return self.__apply_ptmodel(x)

        shape_i = [1, self.shape[1:]]
        size_i = np.prod(shape_i)

        len = int(x.size / size_i)

        for i in range(len):
            x_i = x[size_i * i : size_i * (i + 1)].reshape(shape_i)
            output = x_i
            for layer in self.layers:
                output = layer.apply(output)

        for layer in layers:
            layer.reset()

        return output
