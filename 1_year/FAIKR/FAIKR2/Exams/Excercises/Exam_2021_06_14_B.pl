
win_over(scotland, czechRep):0.3.
win_over(scotland, holland):0.2.

win_one_match(Team) :-
    win_over(Team, _).

query(win_one_match(scotland)).

% It generates 4 worlds:
% 1) T,T ==> 0.3*0.2 = 0.06
% 2) T,F ==> 0.3*(1-0.2) = 0.24
% 3) F,T ==> (1-0.3)*0.2 = 0.14
% 4) F,F ==> (1-0.3)*(1-0.2) = 0.56

% Scotland won in the first 3 so we have a probability:
% (1) + (2) + (3) = 0.44 