from solver.optimize_impl import OptimizeImpl
from solver.sprt_impl import SPRTImpl
from solver.deepcegar_impl import DeepCegarImpl


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


class DeepCegar():
    def __init__(self, max_ref):
        self.max_ref = max_ref

    def solve(self, model, assertion, display=None):
        impl = DeepCegarImpl(self.max_ref)
        impl.solve(model, assertion, display)
