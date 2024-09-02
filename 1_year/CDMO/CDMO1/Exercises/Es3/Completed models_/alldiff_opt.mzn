% alldiff model

include "alldifferent.mzn";

int: n; 
array[1..n] of var 1..n: q; %variables

%column attack constraints
constraint alldifferent(q)::domain;

%diagonal attack constraints
constraint alldifferent([q[i]+i |i in 1..n ])::domain;
constraint alldifferent([q[i]-i | i in 1..n ])::domain;

n = 50;
%objective function 
var int: obj = sum(i in 1..n)(abs(q[i] - i));  

%Search
ann: search_ann;
search_ann = int_search(q, dom_w_deg, indomain_random); 

%for LNS
include "gecode.mzn";

solve 
 :: search_ann 
 :: restart_luby(250) % prevents from local minimal with freq=250
 :: relax_and_reconstruct(q, 85) % relaxes the model by not applying constraints temporarly to 85 variables of q
 minimize obj;

% output["obj = \(obj)\n"];

 output [ if fix(q[j]) == i then "Q" else "." endif ++
                    if j==n then "\n" else "" endif | i,j in 1..n]
                    ++ ["obj = \(obj)\n"];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*
 output [ if fix(q[j]) == i then "Q" else "." endif ++
                    if j==n then "\n" else "" endif | i,j in 1..n]
                    ++ ["obj = \(obj)\n"];
*/                    
