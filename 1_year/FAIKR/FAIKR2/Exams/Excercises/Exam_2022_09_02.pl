p(1):-write('I am solving P.').
r(1):-write('I am solving R.').
q(X):-p(X), r(X).

solve(true) :- !.
solve(write(S)) :- !, write(S),nl.
solve((A,B)) :- !, solve(B),solve(A).
solve(A) :- clause(A,B), solve(B).