line(0,_) :- !,nl.
line(N, C) :-
    write(C),
    N1 is N-1,
    line(N1, C).

square(N,C):- square(N,C,N).
square(_,_,0):- !.
square(N,C,K) :-
    line(N,C),
    K1 is K-1,
    square(N,C, K1).