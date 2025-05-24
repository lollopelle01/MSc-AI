util(_, [], []) :- !.
util(E, [H|T], [[E|H]|Res]) :-  % Correctly prepend E to each subset
    util(E, T, Res).

pow([], [[]]) :- !.  % The power set of an empty list is [[]] (empty set).
pow([H|T], UniqueRes) :-
    pow(T, Res),        % Get power set of the tail
    util(H, Res, R),    % Prepend H to each subset in Res
    append(R, Res, All), % Combine subsets with and without H
    sort(All, UniqueRes).  % Remove duplicates and sort
