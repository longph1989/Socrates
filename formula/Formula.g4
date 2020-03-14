grammar Formula;

formula
    :   conjList EOF
    ;

conjList
    :   conj
    |   conj AND conjList
    ;

conj
    :   X1 EQ LB numList RB             # orig
    |   distFun LP X1 CM X2 RP LT num   # dist
    |   LBL LP Y2 RP EQ num             # label
    |   FAI LP LB numList RB RP         # fair
    |   SI EQ INT AND ineqList          # inCons
    |   SO EQ INT AND ineqList          # outCons
    ;

ineqList
    :   ineq
    |   ineq AND ineqList
    ;

ineq
    :   termList GT FLT
    |   termList EQ FLT
    ;

termList
    :   term
    |   term ADD termList
    ;

term
    :   FLT MUL (X1 | Y1) LB INT RB
    ;

numList
    :   num
    |   num CM numList
    ;

num
    :   INT | FLT
    ;

var
    :   X1 | X2 | Y1 | Y2
    ;

distFun
    :   D0 | D2 | DI
    ;

AND : '&';
CM  : ',';

ADD : '+';
SUB : '-';
MUL : '*';

EQ  : '=';
LT  : '<';
GT  : '>';

LP  : '(';
RP  : ')';
LB  : '[';
RB  : ']';

X1  : 'x1';
X2  : 'x2';
Y1  : 'y1';
Y2  : 'y2';

D0  : 'll_0';
D2  : 'll_2';
DI  : 'll_i';
LBL : 'label';
FAI : 'fair';

SI  : 'inSize';
SO  : 'outSize';

INT : '0' | '-'? [1-9] [0-9]*;
FLT : ('0' | '-'? [1-9] [0-9]*) '.' [0-9]*;

WS  : [ \t\r\n] -> channel(HIDDEN);
