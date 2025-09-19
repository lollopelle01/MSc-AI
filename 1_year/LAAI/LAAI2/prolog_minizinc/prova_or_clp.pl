bill(VARIABLES) :-
    length(VARIABLES, 4),
    VARIABLES = [A,B,C,D],
    VARIABLES ins 10..40,
    A in 20 \/ 40,
    C #= 13 + B,
    D #> C,
    A + B + C + D #= 80,
    label(VARIABLES), !.



    