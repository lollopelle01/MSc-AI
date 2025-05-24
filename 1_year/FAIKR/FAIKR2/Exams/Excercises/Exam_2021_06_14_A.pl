p(a).
p(b).
p(c).

count(P,Res) :-
    findall(P, clause(P,_), L),
    length(L,Res).