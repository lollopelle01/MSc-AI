% Define values for Roman numerals
value('I', 1).
value('V', 5).
value('X', 10).
value('L', 50).
value('C', 100).
value('D', 500).
value('M', 1000).

% Interface
roman2arabic(L, Res) :-
    string_chars(L, Lc),  % Convert string to list of characters
    write(Lc), nl,
    translate(Lc, Res).

% Base case
translate([], 0).

translate([H], R) :- 
    value(H, R).

translate([H1, H2 | T], R) :-  % subtraction cases 
    value(H1, V1),
    value(H2, V2),
    V1 < V2,  % If first value is smaller, it's a subtraction case
    translate(T, Res),
    R is Res + (V2 - V1).

translate([H | T], R) :-  % normal case
    value(H, V),
    translate(T, Res),
    R is Res + V.
