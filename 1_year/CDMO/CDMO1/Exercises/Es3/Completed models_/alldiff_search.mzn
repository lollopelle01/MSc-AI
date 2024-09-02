% alldiff model

include "alldifferent.mzn";

int: n; 
array[1..n] of var 1..n: q; %variables

%column attack constraints
constraint alldifferent(q)::domain;

%diagonal attack constraints
constraint alldifferent([q[i]+i |i in 1..n ])::domain;
constraint alldifferent([q[i]-i | i in 1..n ])::domain;

%search

ann: search_ann;
search_ann = int_search(q, input_order, indomain_min);
%search_ann = int_search(q, input_order, indomain_random);
%search_ann = int_search(q, first_fail, indomain_min);
%search_ann = int_search(q, first_fail, indomain_random);
%search_ann = int_search(q, dom_w_deg, indomain_min); 
%search_ann = int_search(q, dom_w_deg, indomain_random); 

solve :: search_ann  satisfy;