from utils import *


class ContinualImpl():
    def solve(self, model, assertion, display=None):
        lower, upper = model.lower, model.upper

        print("lower = {}".format(lower))
        print("upper = {}".format(upper))

        for i in range(1000):
            x = generate_x(len(lower), lower, upper)
            y = model.apply(x).reshape(-1)

            print("x = {}".format(x))
            print("y = {}".format(y))

        return None