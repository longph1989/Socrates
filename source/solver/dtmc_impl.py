import numpy as np

class DTMCImpl():
    def __init__(self):
        pass

    def __generate_x(self, shape, lower, upper):
        size = np.prod(shape)
        x = np.random.rand(size)

        x = (upper - lower) * x + lower

        return x

    def solve(self, model, assertion, display=None):
        x = self.__generate_x(model.shape, model.lower, model.upper)
        y = np.argmax(model.apply(x), axis=1)[0]
