# Generated from Assertion.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .AssertionParser import AssertionParser
else:
    from AssertionParser import AssertionParser

import autograd.numpy as np
import ast

from .lib_assertions import *
from .lib_functions import *
from utils import *


# This class defines a complete generic visitor for a parse tree produced by AssertionParser.

class AssertionVisitor(ParseTreeVisitor):

    def __init__(self):
        self.init_dict = {}

    def visitImplication(self, ctx:AssertionParser.ImplicationContext):
        vars = list()

        for var in ctx.VAR():
            vars.append(Var(var.getText()))

        pre = self.visitDisjunction(ctx.disjunction(0))
        post = self.visitDisjunction(ctx.disjunction(1))

        return Implication(vars, pre, post, self.init_dict)

    def visitDisjunction(self, ctx:AssertionParser.DisjunctionContext):
        conjs = list()

        for conj in ctx.conjunction():
            conjs.append(self.visitConjunction(conj))

        return Disjunction(conjs)

    def visitConjunction(self, ctx:AssertionParser.ConjunctionContext):
        terms = list()

        for term in ctx.term():
            terms.append(self.visitTerm(term))

        return Conjunction(terms)

    def visitTerm(self, ctx:AssertionParser.TermContext):
        if ctx.TRUE():
            return TrueTerm()
        else:
            if ctx.func(0):
                lhs = self.visitFunc(ctx.func(0))

                if ctx.func(1):
                    rhs = self.visitFunc(ctx.func(1))
                elif ctx.num():
                    rhs = Num(ast.literal_eval(ctx.num().getText()))
            elif ctx.VAR(0):
                if ctx.array():
                    self.init_dict[ctx.VAR(0).getText()] = np.array(ctx.array().getText())
                    return TrueTerm()
                else:
                    vars1 = list()
                    vars1.append(Var(ctx.VAR(0).getText()))

                    idx1 = ast.literal_eval(ctx.INT(0).getText())
                    lhs = Function(wrapped_partial(index, i=idx1), vars1)

                    if ctx.VAR(1):
                        vars2 = list()
                        vars2.append(Var(ctx.VAR(1).getText()))

                        idx2 = ast.literal_eval(ctx.INT(1).getText())
                        rhs = Function(wrapped_partial(index, i=idx2), vars2)
                    elif ctx.num():
                        rhs = Num(ast.literal_eval(ctx.num().getText()))

            op = self.visitOp(ctx.op())

            return GeneralTerm(op, lhs, rhs)

    def visitFunc(self, ctx:AssertionParser.FuncContext):
        vars = list()

        if ctx.D0():
            vars.append(Var(ctx.VAR(0).getText()))
            vars.append(Var(ctx.VAR(1).getText()))
            return Function(d0, vars)

        elif ctx.D2():
            vars.append(Var(ctx.VAR(0).getText()))
            vars.append(Var(ctx.VAR(1).getText()))
            return Function(d2, vars)

        elif ctx.DI():
            vars.append(Var(ctx.VAR(0).getText()))
            vars.append(Var(ctx.VAR(1).getText()))
            return Function(di, vars)

        elif ctx.ARG_MAX():
            vars.append(Var(ctx.VAR(0).getText()))
            return Function(arg_max, vars)

        elif ctx.ARG_MIN():
            vars.append(Var(ctx.VAR(0).getText()))
            return Function(arg_min, vars)

        elif ctx.LIN_INP():
            vars.append(Var(ctx.VAR(0).getText()))
            array = np.array(ast.literal_eval(ctx.array().getText()))
            return Function(wrapped_partial(lin_inp, coefs=array), vars)

        elif ctx.LIN_OUT():
            vars.append(Var(ctx.VAR(0).getText()))
            array = np.array(ast.literal_eval(ctx.array().getText()))
            return Function(wrapped_partial(lin_out, coefs=array), vars)

        else:
            return None

    def visitOp(self, ctx:AssertionParser.OpContext):
        if ctx.GE():
            return Op.GE
        elif ctx.GT():
            return Op.GT
        elif ctx.LE():
            return Op.LE
        elif ctx.LT():
            return Op.LT
        elif ctx.EQ():
            return Op.EQ
        elif ctx.NE():
            return Op.NE


del AssertionParser
