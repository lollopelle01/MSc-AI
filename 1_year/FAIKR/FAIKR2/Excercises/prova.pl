solve(true) :- !.
solve((A, B)) :- !, solve(A), solve(B).
solve(A) :-
    system_predicate(A), !,
    call(A).
solve(A) :- clause(A, B), solve(B).

system_predicate(format(_, _)).
system_predicate(write(_)).
system_predicate(nl).

log(X) :- format("**~w~n", [X]).

p :- q, r.
q :- log('In q!!!!').
r :- log('In r!!!!').
