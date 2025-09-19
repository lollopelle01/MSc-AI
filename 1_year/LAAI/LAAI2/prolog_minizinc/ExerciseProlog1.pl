% ES1: Define a PROLOG predicate lastel that given a list L of integer, find the last element E of the list
lastel([],[]).
lastel([E],E) :- !.
lastel([_|X],Y) :- lastel(X,Y).



% ES2: Define a PROLOG predicate max that given a list L of integer, find the greater element E of the list
max([E],E) :- !.
max([N|L],E) :-
    max(L,E),
    E >= N, !.
max([N|_],N).  % if i've not found a max before, so N must be the max



% ES3: Given a list L1 and a integer number N, write a PROLOG predicate question1(L1, N, L2) where L2 must 
% be the list of elements in L1 that are list with 2 positive value between 1 and 9 which sum is N.
question1([], _, []).
question1([[A,B]|R], N, [[A,B]|S]) :-
    A >= 1, A =< 9,
    B >= 1, B =< 9, 
    Sum is A + B,
    N == Sum,
    !,  % we cut here as the conditions are satisfied ==> sure not to fall in the third predicate
    question1(R, N, S).
question1([_|R], N, S) :-   % if the condition before is not matched L2 is untouched and proceed with the next elem
    question1(R, N, S).



% ES4: Write a Prolog predicate consec that given a list L and an element E, returns the element of 
% L that follows E. If E is the last element or it is not present in the list, the predicate must fail.
consec(E, [E|[H|_]], H) :- !.
consec(E, [_|T], H) :-
    consec(E,T,H).



% ES5: Write a Prolog predicate listPosthat given a list L and an element E, returns the position 
% of the first occurence of E inside L. The first element of the list is considered to be in position 0.
listPosthat([E|_], E, 0) :- !.
listPosthat([_|T], E, Sum) :-
    listPosthat(T, E, S),
    Sum is S + 1.