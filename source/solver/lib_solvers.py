from solver.optimize_impl import OptimizeImpl
from solver.sprt_impl import SPRTImpl
from solver.dtmc_impl import DTMCImpl


class Optimize():
    def solve(self, model, assertion, display=None):
        impl = OptimizeImpl()
        impl.solve(model, assertion, display)


class SPRT():
    def __init__(self, threshold, alpha, beta, delta):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def solve(self, model, assertion, display=None):
        impl = SPRTImpl(self.threshold, self.alpha, self.beta, self.delta)
        impl.solve(model, assertion, display)


class DTMC():
    def __init__(self):
        pass

    def solve(self, model, assertion, display=None):
        impl = DTMCImpl()
        impl.solve(model, assertion, display)
