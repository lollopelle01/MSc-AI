% ES1: Provide a Prolog program that defines a predicate couples(L,C,N) that,
% given a list of integers L, returns a list of couples C that contains all the
% pairs of adjacent numbers in L , where the second element of the
% couple is greater or equal than the first one. The counter N must return
% the number of pairs which have been discarded.
couples([_], [], 0) :- !.
couples([H1|[H2|T]], [[H1,H2]|Res], N) :-
    H2 >= H1,
    !,
    couples([H2|T], Res, N).
couples([_|[H2|T]], Res, N1) :-
    couples([H2|T], Res, N),
    N1 is N + 1.





% ES2: Write a program that defines the predicate sum and prod(L,S,P) that,
% given a list of integers L, computes the product P and sum S of the
% numbers in the list.
sum_and_prod([E], E, E) :- !.
sum_and_prod([H|T], S1, P1) :-
    sum_and_prod(T, S, P),
    S1 is S + H,
    P1 is P * H.




% ES3: Write a Prolog program that defines a predicate selectgreater(L1, L2, R, S),
% that given 2 lists L1 and L2 compares their elements pairwise and returns the
% list R of the greater elements, along with the sum S of all the element in the
% list R. If one of the two lists has more elements than the other, the elements
% in such a list must be included in R.
selectgreater([],[], [], 0) :- !.
selectgreater([], [H|T], [H|Res], S1) :-
    selectgreater([], T, Res, S),
    S1 is S + H.
selectgreater([H|T], [], [H|Res], S1) :-
    selectgreater(T, [], Res, S),
    S1 is S + H.
selectgreater([H1|T1], [H2|T2], [H1|Res], S1) :-
    H1 > H2,
    !,
    selectgreater(T1, T2, Res, S),
    S1 is S + H1.
selectgreater([_|T1], [H2|T2], [H2|Res], S1) :-
    selectgreater(T1, T2, Res, S),
    S1 is S + H2.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ES1: Anton, Beth, Carlos, and Daniel are eating a cake and must divide the 9
% slices between each other. Anthon cooked the cake, so he wants more
% slices than anyone else. Beth has done her workout in the morning, so
% she deserves a treat and wants at least 3 slices. Carlos is on a diet, so
% he will eat less than 3 slices. Daniel wants to feel unique, so he will eat
% a number of slices that is different from anyone else, and at least 1.
% They want to save the remaining slices in the fridge, but the fridge is
% almost full, so only 1 slice can remain.
% Write a CLP or minizinc program to compute how they can divide the
% slices. Please use comments so to make clear what is your reasoning
% and which variables will contain the final results.

cake(VARIABLES) :- 
    length(VARIABLES, 5),
    VARIABLES = [A, B, C, D, R],
    VARIABLES ins 0..9,
    R #=< 1,
    D #\= A, D #\= B, D #\= C,
    D #>= 1,
    C #< 3,
    B #>= 3,
    A #> B, A #> C, A #> D,
    A + B + C + D + R #= 9,
    label(VARIABLES),
    !. % only the first solutution









% ES2: A thief is stealing sculptures in an art gallery. Each sculpture has a
% specific value and a weight. The thief wants to steal sculptures for more
% than 600$ of total value, but can only have 11 kilos of sculptures in the
% sack. The thief must therefore decide what to take and what to leave.
% Write a CLP or Minizinc program to compute which choices thief has,
% and for each choice, what is the final value of the stolen sculptures. The
% sculptures are 4 (A, B, C, D): A weights 10kg and it is worth 500$, B
% weights 4kg and it is worth 600$, C weights 2kg and it is worth 100$, D
% weights 2kg and it is worth 200$.

bag(VARIABLES) :-
    length(VARIABLES, 4),
    VARIABLES = [A,B,C,D],
    VARIABLES ins 0..1,

    Aw #= A * 10,
    Av #= A * 500,
    Bw #= B * 4,
    Bv #= B * 600,
    Cw #= C * 2,
    Cv #= C * 100,
    Dw #= D * 2,
    Dv #= D * 200,

    W #= Aw + Bw + Cw + Dw,
    V #= Av + Bv + Cv + Dv,

    W #=< 11,
    V #> 600,

    labeling([], VARIABLES), nl,
    format('Total_value = ~d $ ~nTotal_weigth = ~d Kg', [V, W]),
    !.