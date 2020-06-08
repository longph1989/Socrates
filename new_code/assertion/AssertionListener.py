# Generated from Assertion.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .AssertionParser import AssertionParser
else:
    from AssertionParser import AssertionParser

# This class defines a complete listener for a parse tree produced by AssertionParser.
class AssertionListener(ParseTreeListener):

    # Enter a parse tree produced by AssertionParser#implication.
    def enterImplication(self, ctx:AssertionParser.ImplicationContext):
        pass

    # Exit a parse tree produced by AssertionParser#implication.
    def exitImplication(self, ctx:AssertionParser.ImplicationContext):
        pass


    # Enter a parse tree produced by AssertionParser#disjunction.
    def enterDisjunction(self, ctx:AssertionParser.DisjunctionContext):
        pass

    # Exit a parse tree produced by AssertionParser#disjunction.
    def exitDisjunction(self, ctx:AssertionParser.DisjunctionContext):
        pass


    # Enter a parse tree produced by AssertionParser#conjunction.
    def enterConjunction(self, ctx:AssertionParser.ConjunctionContext):
        pass

    # Exit a parse tree produced by AssertionParser#conjunction.
    def exitConjunction(self, ctx:AssertionParser.ConjunctionContext):
        pass


    # Enter a parse tree produced by AssertionParser#term.
    def enterTerm(self, ctx:AssertionParser.TermContext):
        pass

    # Exit a parse tree produced by AssertionParser#term.
    def exitTerm(self, ctx:AssertionParser.TermContext):
        pass


    # Enter a parse tree produced by AssertionParser#func.
    def enterFunc(self, ctx:AssertionParser.FuncContext):
        pass

    # Exit a parse tree produced by AssertionParser#func.
    def exitFunc(self, ctx:AssertionParser.FuncContext):
        pass


    # Enter a parse tree produced by AssertionParser#op.
    def enterOp(self, ctx:AssertionParser.OpContext):
        pass

    # Exit a parse tree produced by AssertionParser#op.
    def exitOp(self, ctx:AssertionParser.OpContext):
        pass


    # Enter a parse tree produced by AssertionParser#array.
    def enterArray(self, ctx:AssertionParser.ArrayContext):
        pass

    # Exit a parse tree produced by AssertionParser#array.
    def exitArray(self, ctx:AssertionParser.ArrayContext):
        pass


    # Enter a parse tree produced by AssertionParser#num.
    def enterNum(self, ctx:AssertionParser.NumContext):
        pass

    # Exit a parse tree produced by AssertionParser#num.
    def exitNum(self, ctx:AssertionParser.NumContext):
        pass



del AssertionParser