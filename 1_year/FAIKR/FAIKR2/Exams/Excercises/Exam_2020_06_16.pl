p(X):-q(X), r(X).
p(X):-s(X).
q(X):-t(X).
r(1).
r(2).
r(3).
t(1).
t(2).
s(12).

solve(true, _) :- !.
solve((A,B), L) :-
    !,
    solve(A,LA),
    solve(B,LB),
    append(LA,LB,L).
solve(A, [A|L]) :-
    bagof(X, clause(A,X), L),   % not findall as doesn't unify values in the list
    clause(A,B),
    solve(B,_).
