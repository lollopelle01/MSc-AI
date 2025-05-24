inOddPositions([], []).
inOddPositions([H], [H]).
inOddPositions([H, _|T], [H|Res]) :-
    inOddPositions(T, Res). 

%% If inEvenPosition??
inEvenPositions([], []).
inEvenPositions([_], []).
inEvenPositions([_, H|T], [H|Res]) :-
    inEvenPositions(T, Res).
