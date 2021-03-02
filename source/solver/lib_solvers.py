from solver.optimize_impl import OptimizeImpl
from solver.sprt_impl import SPRTImpl
from solver.deepcegar_impl import DeepCegarImpl
from solver.backdoor_impl import BackDoorImpl


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
    def __init__(self, has_ref, max_ref, ref_typ, max_sus):
        self.has_ref = has_ref
        self.max_ref = max_ref
        self.ref_typ = ref_typ
        self.max_sus = max_sus

    def solve(self, model, assertion, display=None):
        impl = DeepCegarImpl(self.has_ref, self.max_ref, self.ref_typ, self.max_sus)
        return impl.solve(model, assertion, display)


class BackDoor():
    def solve(self, model, assertion, display=None):
        impl = BackDoorImpl()
        return impl.solve(model, assertion, display)
