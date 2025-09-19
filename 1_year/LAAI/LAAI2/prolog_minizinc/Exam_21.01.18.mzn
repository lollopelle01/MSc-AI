include "globals.mzn";

var 9..18: MS;
var 9..18: A;
var 9..18: B;
var 9..18: C;
var 9..18: ME;

constraint alldifferent([MS, A, B, C, ME]);

constraint forall (x in [A, B, C])  (abs(MS-x)>=3);
constraint forall (x in [A, B, C])  (abs(ME-x)>=3);

constraint forall (x in [A, B, C, ME])  (MS < x);
constraint forall (x in [A, B, C, MS])  (ME > x);

constraint abs(A-B)==2;
constraint abs(A-C)==3;

constraint abs(C-B)==1;

constraint B <= 14;

solve satisfy;

