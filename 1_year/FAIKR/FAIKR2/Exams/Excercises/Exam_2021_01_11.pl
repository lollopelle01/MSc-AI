p(X):-q(X), r(X).
p(X):-s(X).
q(X):-t(X).
r(1).
r(2).
r(3).
t(1).
t(2).

verbose(true) :- !.
verbose((A,B)) :- !, verbose(A), verbose(B).
verbose(A) :-
    format("Solving: ~w~n", [A]),
    clause(A,B),
    % format("Solving: ~w~n", [A]),     % NB: format is not correct here as
                                        % A is already assigned by matching
                                        % clauses
    verbose(B).