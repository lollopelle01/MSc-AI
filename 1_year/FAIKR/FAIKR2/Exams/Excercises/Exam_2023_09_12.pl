parent(francesco, federico).
parent(chiara, federico).
parent(francesco, elena).

%% Can't use sort i think :(
% people(Res):-
%     findall(P, (parent(_,P);parent(P,_)), L),
%     sort(L, Res).   % deletes duplicates

%% Can do better but still smart
% member_ex(E, [E|_]).
% member_ex(E, [_|L]) :-
%     member_ex(E,L).

% people(Res):-
%     findall(P, (parent(_,P);parent(P,_)), L),
%     setof(P, member_ex(P,L), Res).

%% BEST
people(L) :- 
    setof(X, Y^(parent(X,Y);parent(Y,X)), L).