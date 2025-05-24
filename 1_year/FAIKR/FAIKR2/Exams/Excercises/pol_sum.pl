sum_ex([], 0) :- !.
sum_ex([(V,_)|T], R) :-
    sum_ex(T, R1),
    R is R1 + V.

delete_ex([], _, []) :-!.
delete_ex([H|T], H, Res) :-
    !,
    delete_ex(T,H,Res).
delete_ex([H|T], E, [H|Res]) :-
    delete_ex(T,E,Res).

pol_sum([], L, L) :- !.  
pol_sum([(C, G)|T], L, [(S, G)|FinalRes]) :-
    findall((X, G), member((X, G), L), L2),  
    append([(C, G)], L2, Ltot),  
    sum_ex(Ltot, S),  
    delete_ex(L, (_, G), LFiltered),  
    pol_sum(T, LFiltered, FinalRes).