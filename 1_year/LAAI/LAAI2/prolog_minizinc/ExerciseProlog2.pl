% ES1: Compute the absolute value of a number
abs(X, X) :- X >= 0, !.
abs(X, Y) :- Y is -X.



% ES2: Compute the factorial of a number
fact(0, 1) :- !.
fact(X, Y) :- 
    X_new is X-1,
    fact(X_new, Y_new),
    Y is X * Y_new.
    % format('X:~d, Y:~d ~n', [X,Y]).



% ES3: Compute the greatest common divisor
gcd(X,0,X) :- !.
gcd(X,Y,Z) :-
    K is X mod Y,
    gcd(Y, K, Z).



% ES4: Find the last element of a list
lastel([E], E) :- !.
lastel([_|L], E) :-
    lastel(L,E).



% ES5: Check if a list is a sublist of another list
isSub([], _) :- !.              % empty list is always sublist
isSub([X|T1], [X|T2]) :-        
    isSub(T1, T2).
isSub(L, [_|T2]) :-             
    isSub(L, T2).