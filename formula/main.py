import sys
from antlr4 import *
from FormulaLexer import FormulaLexer
from FormulaParser import FormulaParser
from FormulaPrtVisitor import FormulaPrtVisitor

def main(argv):
    while True:
        text = InputStream(input(">"))
        lexer = FormulaLexer(text)
        stream = CommonTokenStream(lexer)
        parser = FormulaParser(stream)

        tree = parser.formula()
        FormulaPrtVisitor().visitFormula(tree)

if __name__ == '__main__':
    main(sys.argv)
