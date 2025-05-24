filter([], []) :- !.
filter([p(X)|T], [p(Y)|Res]) :-
    number(X), !,
    Y is X + 1,
    filter(T, Res).
filter([H|T], [H|Res]) :-
    filter(T, Res).
    