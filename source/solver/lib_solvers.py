from solver.optimize_impl import OptimizeImpl
from solver.sprt_impl import SPRTImpl
from solver.refinement_impl import RefinementImpl
from solver.backdoor_impl import BackDoorImpl
from solver.backdoor_repair_impl import BackDoorRepairImpl
from solver.dtmc_impl import DTMCImpl
from solver.dtmc_rnn import DTMCImpl_rnn
from solver.verifair_impl import VeriFairimpl
from solver.causal_impl import CausalImpl
from solver.continual_impl1 import ContinualImpl1
from solver.continual_impl2 import ContinualImpl2
from solver.continual_impl3 import ContinualImpl3
from solver.backdoor_detect_impl import BackdoorDetectImpl

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


class Refinement():
    def __init__(self, has_ref, max_ref, ref_typ, max_sus):
        self.has_ref = has_ref
        self.max_ref = max_ref
        self.ref_typ = ref_typ
        self.max_sus = max_sus

    def solve(self, model, assertion, display=None):
        impl = RefinementImpl(self.has_ref, self.max_ref, self.ref_typ, self.max_sus)
        return impl.solve(model, assertion, display)


class BackDoor():
    def solve(self, model, assertion, display=None):
        impl = BackDoorImpl()
        return impl.solve(model, assertion, display)


class BackDoorRepair():
    def solve(self, model, assertion, display=None):
        impl = BackDoorRepairImpl()
        return impl.solve(model, assertion, display)


class DTMC():
    def __init__(self):
        pass
    def solve(self, model, assertion, display=None):
        impl = DTMCImpl()
        impl.solve(model, assertion, display)

class DTMC_rnn():
    def __init__(self):
        pass
    def solve(self, model, assertion, display=None):
        impl = DTMCImpl_rnn()
        impl.solve(model, assertion, display)

class VeriFair():
    def __init__(self):
        pass
    def solve(self, model, assertion, display=None):
        impl = VeriFairimpl()
        impl.solve(model, assertion, display)

class Causal():
    def __init__(self):
        pass
    def solve(self, model, assertion, display=None):
        impl = CausalImpl()
        impl.solve(model, assertion, display)

class Continual():
    def __init__(self):
        pass
    def solve(self, models, assertion, display=None):
        if assertion['dataset'] == 'acasxu':
            impl = ContinualImpl1()
        elif assertion['dataset'] == 'mnist' or assertion['dataset'] == 'cifar10':
            impl = ContinualImpl2()
        elif assertion['dataset'] == 'census':
            impl = ContinualImpl3()
        else:
            assert False

        impl.solve(models, assertion, display)

class BackdoorDetect():
    def __init__(self):
        pass
    def solve(self, model, assertion, display=None):
        impl = BackdoorDetectImpl()
        impl.solve(model, assertion, display)
