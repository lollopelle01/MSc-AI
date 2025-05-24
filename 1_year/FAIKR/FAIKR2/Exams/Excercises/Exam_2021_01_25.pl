p(X):-q(X), r(X).
p(X):-s(X).
q(X):-t(X).
r(1).
r(2).
r(3).
t(1).
t(2).

verbose(true) :- !.
verbose((A,B)) :-
    !,
    verbose(A),
    verbose(B).
verbose(A) :-
    clause(A,B),
    verbose(B),
    format("Solved: ~w~n",[A]).