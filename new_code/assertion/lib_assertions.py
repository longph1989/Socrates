import enum


class Assertion:
    def __init__(self, impls):
        self.impls = impls


class Implication:
    def __init__(self, vars, pre, post):
        self.vars = vars
        self.pre = pre
        self.post = post


class Disjunction:
    def __init__(self, conjs):
        self.conjs = conjs


class Conjunction:
    def __init__(self, terms):
        self.terms = terms


class TrueTerm:
    pass


class GeneralTerm:
    def __init__(self, type, lhs, rhs):
        self.type = type
        self.lhs = lhs
        self.rhs = rhs


class Function:
    def __init__(self, func, vars):
        self.func = func
        self.vars = vars


class Var:
    def __init__(self, name):
        self.name = name


class Op(enum.Enum):
   GE = 1
   GT = 2
   LE = 3
   LT = 4
   EQ = 5
   NE = 6
