% Assegnazione delle variabili
var 10..40: A;
var 10..40: B;
var 10..40: C;
var 10..40: D;

% Vincoli
constraint A in {20, 40};            % A in 20 \/ 40
constraint C = 13 + B;              % C #= 13 + B
constraint D > C;                   % D #> C
constraint A + B + C + D = 80;      % A + B + C + D #= 80

% Risoluzione
solve satisfy;
