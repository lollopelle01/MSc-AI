%%% ES 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

match(bologna,atalanta, 1,0).
match(bologna, genoa,1,1).
match(salernitana,bologna,1,2) .
match(cagliari,bologna,2,1).

match_won(T, Res) :-
    match_won_out_home(T, L1),
    match_won_in_home(T, L2),
    append(L1, L2, Res).

match_won_in_home(T, Res) :-
    findall(match(T, OT, H, NH), (match(T, OT, H, NH), H>NH), Res).

match_won_out_home(T, Res) :-
    findall(match(OT, T, H, NH), (match(OT, T, H, NH), H<NH), Res).

%%% ES 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p :- q,r.
q :- log('In q!!!').
r :- log('In r!!!').

log(A) :- format("++++\'~w\'++++~n", [A]).
syste_pred(log(_)).

solve(true) :- !.
solve((A,B)) :- !, solve(A), solve(B).
solve(A) :- clause(A,B), syste_pred(B), !, B.
solve(A) :- clause(A,B), solve(B).