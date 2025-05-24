% %% Mia versione (CON INTERFACCIA)
% split(L, S, E, R) :-        % interface
%     S >= 0, E>=S, 
%     length(L, N), N>S,
%     split(L, S, E, R, 1).
% % split(_, 0, 0, [], _) :- !.  % handled also by the next case
% split(_, K, K, [],_) :- !.
% split([H|T], S, E, [H|Res], Pos) :- 
%     S =< Pos, E > Pos, !,
%     Pos1 is Pos + 1,
%     S1 is S + 1,
%     split(T, S1, E, Res, Pos1).
% split([_|T], S, E, Res, Pos) :-
%     Pos1 is Pos + 1,
%     split(T, S, E, Res, Pos1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Mia versione (SENZA INTERFACCIA)
check(L, S, E, _) :-            % general checks
    S>=0, E>=S, 
    length(L, N), S=<N, E=<N.
split(_, K, K, []) :- !.        % handles also the 0s case
split([H|T], 0, E, [H|Res]) :-  % add case
    check([H|T], 0, E, [H|Res]), !,
    E1 is E-1,
    split(T, 0, E1, Res).
split([H|T], S, E, Res) :-      % discard case
    check([H|T], S, E, Res),
    S1 is S-1,
    E1 is E-1,
    split(T, S1, E1, Res).
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Versione giorgio
% split([], _, _, []) :- !.
% split(_, H, H, []) :- !.
% split(T, S, _, _) :- length(T, N), S>N, !, fail.
% split(T, S, E, R) :- length(T, N), E>N, !, split(T, S, N, R).
% split([H|T], 1, S, [H|R]) :- 0=<S, !, S1 is S-1, split(T, 1, S1, R).
% split([_|T], G, S, R) :- G=<S, G1 is G-1, S1 is S-1, split(T, G1, S1, R).