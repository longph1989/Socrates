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
    def __init__(self, has_ref, max_ref, ref_typ, has_tig, max_tig):
        self.has_ref = has_ref
        self.max_ref = max_ref
        self.ref_typ = ref_typ

        self.has_tig = has_tig
        self.max_tig = max_tig

    def solve(self, model, assertion, display=None):
        impl = DeepCegarImpl(self.has_ref, self.max_ref, self.ref_typ, self.has_tig, self.max_tig)
        impl.solve(model, assertion, display)
