p(1).
p(2).
r(1).
q(X):-p(X).

solve(true,_,0) :- !.
solve((A,B), P, N) :-
    !,
    solve(A,P,NA),
    solve(B,P,NB),
    N is NA+NB.
solve(P,P,N1) :- 
    !,      % important in order not to evaluate the P twice
    clause(P,B), solve(B,P,N),
    N1 is N+1.
solve(A,P,N) :-
    %% !, is useless ==> to avoid the "; false" put the ! after the query
    clause(A,B), solve(B,P,N).

%% PROF SOL : he also has the problem of "; false" 
%% which does not appear in expected output :)
% p(1).
% p(2).
% r(1).
% q(X) :-p(X).
% solve(true, _Pred, 0) :-!.
% solve((A,B), Pred, Count):-!,solve(A, Pred, C1),solve(B, Pred, C2),Count is C1+C2.
% solve(Pred, Pred, Count):-!,clause(Pred, Body),solve(Body, Pred, CountBody),Count is CountBody + 1.
% solve(Goal, Pred, Count):-!,clause(Goal, Body),solve(Body, Pred, Count).
