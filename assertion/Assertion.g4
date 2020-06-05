grammar Assertion;


assertion
      :   implication (OR implication)* EOF
      ;

implication
      :   LP FA VAR (CM VAR)* DOT disjunction IMP disjunction RP
      ;

disjunction
      :   conjunction (OR conjunction)*
      ;

conjunction
      :   term (AND term)*
      ;

term
      :   func GE num
      |   func GT num
      |   func LE num
      |   func LT num
      |   func EQ num
      |   func NE num
      |   VAR LB INT RB GE num
      |   VAR LB INT RB GT num
      |   VAR LB INT RB LE num
      |   VAR LB INT RB LT num
      |   VAR LB INT RB EQ num
      |   VAR LB INT RB NE num
      |   VAR LB INT RB EQ VAR LB INT RB
      |   VAR EQ array
      ;

func
      :   name LP (VAR | num) (CM (VAR | num))* RP
      |   name LP func RP
      ;

name
      :   'd_0'
      |   'd_2'
      |   'd_i'
      |   'm'
      |   'la'
      |   'li'
      ;

array
      :   LB num (CM num)* RB
      ;

num
      :   INT
      |   FLT
      ;


VAR   :   [a-zA-Z_][a-zA-Z0-9_]* ;
INT   :   '0' | '-'? [1-9][0-9]*;
FLT   :   ('0' | '-'? [1-9][0-9]*) '.' [0-9]*;

LP    :   '(' ;
RP    :   ')' ;
LB    :   '[' ;
RB    :   ']' ;
CM    :   ',' ;
FA    :   '\\A' ;
DOT   :   '.' ;

GE    :   '>=' ;
GT    :   '>' ;
LE    :   '<=' ;
LT    :   '<' ;
EQ    :   '=' ;
NE    :   '!=' ;

OR    :   '\\/' ;
AND   :   '/\\' ;
IMP   :   '=>' ;

WS    :   [ \t\r\n]+ -> skip ;
