%% Es 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filter([],_,[]) :- !.
filter([H|T], L, [H|Res]) :-
    findall(H, member(H,L), Occs),
    length(Occs, N), N>1, !,
    filter(T, L, Res).
filter([_|T], L, Res) :-
    filter(T, L, Res).

%% Es 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example
p :- a,b.
a:-break.
b.

solve(true) :- !.
solve((A,B)) :- !, solve(A), solve(B).
solve(break) :-
    !,
    format("Break: do you want to continue? (yes/no).~n"),
    read(yes).
solve(A) :- clause(A,B), solve(B).
