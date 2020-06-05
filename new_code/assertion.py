class Assertion:
    def __init__(self, impls):
        self.impls = impls


class Implication:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post


class Disjunction:
    def __init__(self, conjs):
        self.conjs = conjs


class Conjunction:
    def __init__(self, terms):
        self.terms = terms


class Term:
    def __init__(self, type, func):
        self.type = type
        self.func = func
