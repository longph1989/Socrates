# Generated from Assertion.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .AssertionParser import AssertionParser
else:
    from AssertionParser import AssertionParser

from assertion import *

# This class defines a complete generic visitor for a parse tree produced by AssertionParser.

class AssertionVisitor(ParseTreeVisitor):

    def visitAssertion(self, ctx:AssertionParser.AssertionContext):
        impls = list()

        for impl in ctx.implication():
            impls.append(self.visitImplication(impl))

        return Assertion(impls)

    def visitImplication(self, ctx:AssertionParser.ImplicationContext):
        vars = list()

        for var in ctx.VAR():
            vars.append(Var(var.getText()))

        pre = self.visitDisjunction(ctx.disjunction(0))
        post = self.visitDisjunction(ctx.disjunction(1))

        return Implication(vars, pre, post)

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
            return Term('True', None, None)
        else:
            return Term('False', None, None)


del AssertionParser
