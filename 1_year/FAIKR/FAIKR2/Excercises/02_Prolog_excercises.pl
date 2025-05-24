% Define a Prolog program that receives in input a number N, and
% print all the numbers between 1 and N.
print_1_N(1) :-
    write(1), nl, !.
print_1_N(N) :-
    % N > 1,
    N1 is N-1,
    print_1_N(N1),
    write(N), nl.



% Define a Prolog program that computes the Fibonacci number.
fibonacci(0,0) :- !.
fibonacci(1,1) :- !.
fibonacci(N, R) :-
    N > 1,
    N1 is N-1,
    N2 is N-2,
    fibonacci(N1,R1),
    fibonacci(N2,R2),
    R is R1 + R2.



% Write a predicate about a number N, that is true if N is prime
is_prime(N) :-  % interface for positive numbers
    N > 1, !,
    N1 is N - 1,
    is_prime(N,N1), !.
is_prime(N) :-  % interface for negative numbers
    N_pos is -N,
    N1 is N_pos - 1,
    is_prime(N_pos,N1), !.
is_prime(_, 1) :- !.    
is_prime(N, C) :-
    C > 1,
    % format('N=~d C=~d ~n', [N,C]),
    not(0 is N mod C),
    C1 is C - 1,
    is_prime(N,C1).



% Write a predicate that, given a number N, prints out all the prime 
% numbers between 2 and N.
print_primes(2) :- write(2), nl, !.
print_primes(N) :-
    N > 2,
    is_prime(N), !,
    N1 is N - 1,
    print_primes(N1),
    write(N), nl.
print_primes(N) :-
    N > 2,
    N1 is N - 1,
    print_primes(N1).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define a predicate isList/1 that is true only if the argument is a list.
% Starting from the recursive definition, T is a list if:
% • T is an empty list, or
% • T is a non empty list, and its tail is a list
isList([]) :- !.
isList([_|T]) :- isList(T).




% Define a predicate member/2 that is true if the first argument is an element 
% of the list passed as a second argument.
member(E, [E|_]) :- !.  % PS: remember that by applying the cut here you won't see correctly 
                        % the output of member(X, [1,4,2,...]).
member(E, [_|T]) :-
    member(E, T).



% Define a predicate length/2 that takes as first argument a list, and the second 
% argument is the number of elements contained in the list.
% NB: renamed to length_ex as the static method "length" already existed
length_ex([],0).
length_ex([_|T], R) :-
    length_ex(T,R1),
    R is R1 + 1.




% Define a predicate append/3 that takes as first and second arguments two lists, 
% and the third argument is the list obtained by concatenating the two lists
% NB: list must be built from the beginning!!
append_ex([], L, L).
append_ex([H1|T1], L, [H1|T_new]) :-
    append_ex(T1, L, T_new).



% Define a predicate deleteFirstOccurence/3 that takes as first and second argument 
% an element and alist respectively, and the third argument is the list without
% the first occurrence of the element (without the term that unifies with the element).
deleteFirstOccurence(E, [E|L], L) :- !.
deleteFirstOccurence(E, [H|L], [H|Res]) :-
    deleteFirstOccurence(E, L, Res).




% Define a predicate deleteAllOccurences/3 that takes as first and second argument an 
% element and a list respectively, and the third argument is the list without all
% the terms that unify with the element.
deleteAllOccurences(_, [], []) :- !.
deleteAllOccurences(E, [E|T], T_new) :-
    deleteAllOccurences(E, T, T_new), !.
deleteAllOccurences(E, [H|T], [H|T_new]) :-
    deleteAllOccurences(E, T, T_new).




% Define a predicate reverse/2 that takes as first and second argument two lists, where 
% one is the reversed of the second.
reverse([], []) :- !.
% reverse([E], [E]) :- !.     %% not necessary ==> handle in the next one when T=[]
reverse([H|T], Res) :-
    reverse(T, Res1),
    append(Res1, [H], Res).




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GIVEN EXCERCISES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) Write a predicate that given a list, it returns the last element.
last_el([E|[]], E) :- !.
last_el([_|T], E) :-
    last_el(T,E).




% 2) Write a predicate that given two lists L1 and L2, returns true if and only if L1 
% is a sub-list of L2.
is_sublist([], _) :- !.
is_sublist([H1|T1], [H1|T2]) :-
    is_sublist(T1,T2), !.
is_sublist([H1|T1], [_|T2]) :-
    is_sublist([H1|T1],T2).




% 3) Write a predicate that returns true if and only if a list is a palindrome.
is_equal_list([], []).
is_equal_list([H|T1],[H|T2]) :-
    is_equal_list(T1,T2).
is_palindrome(L) :- 
    reverse(L, L1),
    is_equal_list(L,L1).




% 4) Write a predicate that, given a list (possibly with repeated elements), returns 
% a new list with repeated elements.
which_repeated([],[]).
which_repeated([H|T], [H|Res]) :-    % interface
    member(H,T), !,
    deleteAllOccurences(H, T, T_clear),
    which_repeated(T_clear, Res).
which_repeated([_|T], Res) :-
    which_repeated(T, Res).




% 5) Write a predicate that given a term T and a list L, counts the number of
% occurrences of T in L.
count_occurrences(E, L, Res):-
    length(L,N),
    deleteAllOccurences(E, L, L1),
    length(L1, N1),
    Res is N-N1.




% 6) Write a predicate that, given a list, returns a new list obtained by flattening the first list. 
% Example: given the list [1,[2,3,[4]],5,[6]] the predicate should return the list [1,2,3,4,5,6].
flatten([],[]).
flatten([H|T], [H|Res]) :-
    not(isList(H)), !,
    flatten(T, Res).
flatten([H|T], R) :-
    flatten(H, H_flat),
    flatten(T, Res),
    append(H_flat, Res, R).




% 7) Write a predicate that given a list, returns a new list that is the first one, but ordered.

% BUBBLE SORT implementation:
% 1. Boolean flag for the main predicate
to_be_sorted([_|[]]) :- !.
to_be_sorted([H1|[H2|T]]) :-
    H1 =< H2,
    to_be_sorted([H2|T]).

% 2. Predicate for a binary swap
sort_list([],[]) :- !.
sort_list([E],[E]) :- !.
sort_list([H1|[H2|T]], [H2|Res]) :-
    H2 < H1, !,
    sort_list([H1|T], Res).
sort_list([H1|[H2|T]], [H1|Res]) :-
    sort_list([H2|T], Res).

% 3. General predicate that makes binary swaps all over the list until it's sorted
sort_ex(L,Res) :-
    sort_list(L, Res_temp),
    \+ to_be_sorted(Res_temp), !,
    sort_ex(Res_temp, Res).
sort_ex(L,Res) :-
    sort_list(L, Res).