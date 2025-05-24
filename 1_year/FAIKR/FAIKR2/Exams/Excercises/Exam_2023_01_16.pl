member_aux(E, [E|_]).
member_aux(E, [_|T]) :-
    member_aux(E,T).

filter(L1, L2, L3) :-
    setof(E, (member_aux(E,L2),member_aux(E,L1)), L3).