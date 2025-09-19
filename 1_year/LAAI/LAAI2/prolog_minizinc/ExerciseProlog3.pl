%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROLOG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ES1: Create a Prolog predicate called range(N, M, Res) that returns a list containing all the 
% integer numbers from N to M.
range(N,N,[N]) :- !.
range(N,M, [N|Res]) :-
    N1 is N + 1,
    range(N1,M,Res).

% % NOTE: going from M to N gives a bad formatted list as prolog works in 1 direction, you need append
% range(M,M,[M]) :- !.
% range(N,M, Res) :-
%     M1 is M - 1,
%     range(N,M1,Res1),
%     append(Res1, [M], Res).




% ES2: Write a Prolog predicate, called even(List, Res), that given a list, checks all the numerical
% elements and only returns the even ones, leaving all the non-numerical elements in the list.
% If the list contains sub-lists, they are considered as a non-numerical element and therefore
% have to be in the resulting list, no matter their content.
even([], []).
even([H|T], [H|Res]) :-
    number(H),
    0 is H mod 2,
    !,  % conditions passed, we know we won't go in the third predicate
    even(T,Res).
even([H|T], Res) :-
    number(H),
    !,  % conditions passed, we know we won't go in the fourth predicate
    even(T,Res).
even([H|T], [H|Res]) :-
    even(T,Res).