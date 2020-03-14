# Generated from Formula.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .FormulaParser import FormulaParser
else:
    from FormulaParser import FormulaParser

from FormulaVisitor import FormulaVisitor

# This class defines a complete generic visitor for a parse tree produced by FormulaParser.

class FormulaPrtVisitor(FormulaVisitor):

    def __init__(self):
        self.inSize = 0
        self.outSize = 0
        self.inCons = False
        self.outCons = False
        self.array = None
        self.local = False


    # Visit a parse tree produced by FormulaParser#formula.
    def visitFormula(self, ctx:FormulaParser.FormulaContext):
        self.visit(ctx.conjList())

        if self.local:
            print('"robustness" : "local"')
        elif self.inSize == 0 and self.outSize == 0:
            print('"robustness" : "global"')


    # Visit a parse tree produced by FormulaParser#conjList.
    def visitConjList(self, ctx:FormulaParser.ConjListContext):
        self.visit(ctx.conj())
        if ctx.conjList():
            self.visit(ctx.conjList())


    # Visit a parse tree produced by FormulaParser#orig.
    def visitOrig(self, ctx:FormulaParser.OrigContext):
        self.local = True

        print('"origin" : "[', end='')
        self.visit(ctx.numList())
        print(']",')


    # Visit a parse tree produced by FormulaParser#dist.
    def visitDist(self, ctx:FormulaParser.DistContext):
        print('"distance" : ', end='')
        self.visit(ctx.distFun())
        print(',')

        print('"eps" : ', end='')
        self.visit(ctx.num())
        print(',')


    # Visit a parse tree produced by FormulaParser#label.
    def visitLabel(self, ctx:FormulaParser.LabelContext):
        print('"target" : ', end='')
        self.visit(ctx.num())
        print(',')


    # Visit a parse tree produced by FormulaParser#fair.
    def visitFair(self, ctx:FormulaParser.FairContext):
        print('"fairness" : "[', end='')
        self.visit(ctx.numList())
        print(']",')


    # Visit a parse tree produced by FormulaParser#inCons.
    def visitInCons(self, ctx:FormulaParser.InConsContext):
        self.inSize = int(str(ctx.INT()))
        self.inCons = True

        print('"in_cons" : [')
        self.visit(ctx.ineqList())
        print('],')

        self.inCons = False


    # Visit a parse tree produced by FormulaParser#outCons.
    def visitOutCons(self, ctx:FormulaParser.OutConsContext):
        self.outSize = int(str(ctx.INT()))
        self.outCons = True

        print('"out_cons" : [')
        self.visit(ctx.ineqList())
        print('],')

        self.outCons = False


    # Visit a parse tree produced by FormulaParser#ineqList.
    def visitIneqList(self, ctx:FormulaParser.IneqListContext):
        self.visit(ctx.ineq())
        if ctx.ineqList():
            print(',')
            self.visit(ctx.ineqList())
        else:
            print()


    # Visit a parse tree produced by FormulaParser#ineq.
    def visitIneq(self, ctx:FormulaParser.IneqContext):
        s = -float(str(ctx.FLT()))

        if self.inCons:
            self.array = ['0'] * (self.inSize + 1)
            self.array[self.inSize] = str(s)
        elif self.outCons:
            self.array = ['0'] * (self.outSize + 1)
            self.array[self.outSize] = str(s)

        print('\t{')
        if ctx.GT():
            print('\t\t"type" : "ineq",')
        elif ctx.EQ():
            print('\t\t"type" : "eq",')
        print('\t\t"coef" : "[', end='')

        self.visit(ctx.termList())
        s = ', '.join(self.array)
        print(s, end='')

        print(']"')
        print('\t}', end='')


    # Visit a parse tree produced by FormulaParser#termList.
    def visitTermList(self, ctx:FormulaParser.TermListContext):
        self.visit(ctx.term())
        if ctx.termList():
            self.visit(ctx.termList())


    # Visit a parse tree produced by FormulaParser#term.
    def visitTerm(self, ctx:FormulaParser.TermContext):
        index = int(str(ctx.INT()))
        self.array[index] = str(ctx.FLT())


    # Visit a parse tree produced by FormulaParser#numList.
    def visitNumList(self, ctx:FormulaParser.NumListContext):
        self.visit(ctx.num())
        if ctx.numList():
            print(',', end='')
            self.visit(ctx.numList())


    # Visit a parse tree produced by FormulaParser#num.
    def visitNum(self, ctx:FormulaParser.NumContext):
        if ctx.INT():
            print(str(ctx.INT()), end='')
        elif ctx.FLT():
            print(str(ctx.FLT()), end='')


    # Visit a parse tree produced by FormulaParser#var.
    def visitVar(self, ctx:FormulaParser.VarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FormulaParser#distFun.
    def visitDistFun(self, ctx:FormulaParser.DistFunContext):
        if ctx.D0():
            print('"ll_0"', end='')
        elif ctx.D2():
            print('"ll_2"', end='')
        elif ctx.DI():
            print('"ll_i"', end='')



del FormulaParser

# Test: x1 = [1,2,3.4] & ll_i(x1, x2) < 2.3 & label(y2) = 2 & fair([1,2,5])
# Test: ll_2(x1, x2) < 2.3 & fair([1,2,5])
# Test: inSize = 3 & 1.1 * x1[0] + -1.1 * x1[1] > 1.2 & -1.2 * x1[2] = -1.3 & outSize = 5 & 1.0 * y1[1] = 2.0
