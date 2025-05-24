%% Define a predicate isList/1 that is true only if the argument is a list.
isList([]).
% isList([H|T]) :- isList(T).
isList([_|T]) :- isList(T).



%% Define a predicate member/2 that is true if the first argument is an 
%% element of the list passed as a second argument.
member(E, [E|_]).
member(E, [_|T]) :- member(E,T).



%% Define a predicate length/2 that takes as first argument a list, and 
%% the second argument is the number of elements contained in the list.
length_predicate([],0).
length_predicate([_|T],N) :- 
    N1 is N-1,
    length_predicate(T,N1).



%% Define a predicate append/3 that takes as first and second arguments 
%% two lists, and the third argument is the list obtained by concatenating 
%% the two lists.
append([], L1, L1).
append([H |Rest1], L2, [H | NewTail]) :-
    append(Rest1, L2, NewTail).


%% Define a predicate deleteFirstOccurence/3 that takes as first and second 
%% argument an element and a list respectively, and the third argument is the 
%% list without the first occurrence of the element (without the term that 
%% unifies with the element).
deleteFirstOccurence(_, [], []). % Just in case we pass an empty list
deleteFirstOccurence(E, [E|T], T).
deleteFirstOccurence(E, [H|T], [H|Res]) :-
    deleteFirstOccurence(E, T, Res).
% NOTE:     There could be problems with backtracking, but we don't see them
%           since prolog tries the second clause first when the element to 
%           remove matches the head of the list. Once it succeeds, it doesn't 
%           backtrack to try other clauses.


%% Define a predicate deleteAllOccurences/3 that takes as first and second 
%% argument an element and a list respectively, and the third argument is the 
%% list without all the terms that unify with the element.
deleteAllOccurences(_, [], []). % Just in case we pass an empty list
deleteAllOccurences(E, [E|T], Res) :- deleteAllOccurences(E, T, Res),!. % add cut operator so it doesn't backtrack
deleteAllOccurences(E, [H|T], [H|T1]) :- deleteAllOccurences(E, T, T1).



%% Define a predicate reverse/2 that takes as first and second argument two lists, 
%% where one is the reversed of the second.

% Works but bad format ([1,2,3,4,5] becomes [[[[5, 4]|3]|2]|1] instead of [5, 4, 3, 2, 1])
% reverse([X,Y], [Y,X]).
% reverse([H|T], [Res|H]) :-
%     reverse(T, Res).

% Solution (uses append)
reverse([],[]). 
reverse([H|T], Result) :-
    reverse(T, Partial), 
    append(Partial, [H], Result).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Lists in Prolog – some exercises %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOTE %%
% format("~w is equal to ~w ~n", [E1, E2]),   % ~w for a value in the curresponding list
%                                             % ~n for a newline


%% 1) Write a predicate that given a list, it returns the last element.
last_elem([E], E).
last_elem([_|T], E) :-
    last_elem(T, E), !.




%% 2) Write a predicate that given two lists L1 and L2, returns true if 
%% and only if L1 is a sub-list of L2.

% Helper predicate: true if L1 is a prefix of L2
prefix([], _).
prefix([X|L1], [X|L2]) :- 
    prefix(L1, L2).
% Main predicate: true if L1 is a sublist of L2
is_sublist(L1, L2) :- 
    prefix(L1, L2).  % Base case: L1 is a prefix of L2
is_sublist(L1, [_|L2]) :- 
    is_sublist(L1, L2).  % Recursively check the rest of L2




%% 3) Write a predicate that returns true if and only if a list is a palindrome.
is_palindrome(L) :-
    reverse(L, R),
    L==R.




%% 4) Write a predicate that, given a list (possibly with repeated elements), 
%% returns a new list with repeated elements.
count(_, [], 0).
count(Elem, [Elem|Tail], Count) :-
    count(Elem, Tail, RestCount),
    Count is RestCount + 1, !.
count(Elem, [_|Tail], Count) :-
    count(Elem, Tail, Count).

only_repeated(List, Result) :-
    only_repeated(List, [], Result).    % Initialize the "seen" list

only_repeated([], _, []).
only_repeated([H|T], Seen, [H|Rest]) :-
    count(H, T, N),
    N > 0,                  % H occurs one or more times in the tail --> 2 or more in the whole list 
    \+ member(H, Seen),     % Never seen H before
    !,                      % Avoid result loop
    only_repeated(T, [H|Seen], Rest).
only_repeated([_|T], Seen, Rest) :- 
    only_repeated(T, Seen, Rest).




%% 5) Write a predicate that given a term T and a list L, counts the 
%% number of occurrences of T in L

% ... count/3 defined before ... %




%% 6) Write a predicate that, given a list, returns a new list obtained 
%% by flattening the first list. 
%% Example:     given the list [1,[2,3,[4]],5,[6]] the predicate should 
%%              return the list [1,2,3,4,5,6].
flatten([], []).
flatten([H|T], FlatList) :-
    is_list(H),  % If the head is a list, flatten it
    flatten(H, FlatHead), !,
    flatten(T, FlatTail),
    append(FlatHead, FlatTail, FlatList).
flatten([H|T], [H|FlatTail]) :- % If the head is not a list comes here
    flatten(T, FlatTail), !.




%% 7) Write a predicate that given a list, returns a new list that is the first 
%% one, but ordered (INCREASING).

%% Predicate to check if an element is the minimum of a list
is_min(_, []).  % Base case: any element is the minimum of an empty list
is_min(E, [H|T]) :-
    E =< H,    % E must be smaller or equal to H
    is_min(E, T).  % Continue checking the rest of the list

%% Main sorting predicate (insertion sort-like)
mine_sort([], []).  % Base case: an empty list is already sorted
mine_sort([H|T], Res) :- 
    sort_cached([H|T], [], Res),  % Initialize sorting with an empty cache
    !.

%% If the head of the list is the minimum, add it to the result
sort_cached([H|T], Cache, [H|Res]) :-
    is_min(H, T),  % Check if H is the smallest in the rest of the list
    sort_cached(T, Cache, Res).  % Continue sorting the tail

%% If the list is reduced to one element, sort the cache and add the element
sort_cached([H], Cache, [H|Res]) :-
    mine_sort(Cache, Res).  % Sort the remaining elements in the cache

%% If H is not the smallest, move it to the cache and continue
sort_cached([H|T], Cache, Res) :-
    \+ is_min(H, T),  % If H is not the smallest, move it to the cache
    sort_cached(T, [H|Cache], Res).  % Add H to the cache and continue
