a(X):-b(X), c(X).
a(X):-Y is X+1, b(Y), c(Y).
a(X):-Y is X-1, b(Y), c(Y).

how_many(Pred, Num_of_clauses):-
    findall(X, clause(Pred, X), L),
    length(L, Num_of_clauses).