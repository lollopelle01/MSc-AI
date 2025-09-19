set of int: SLICES = 0..9;

var SLICES: A;
var SLICES: B;
var SLICES: C;
var SLICES: D;

constraint A > B;
constraint A > C;
constraint A > D;

constraint B >= 3;

constraint C < 3;

constraint D != A;
constraint D != B;
constraint D != C;
constraint D > C;

constraint A + B + C + D >= 8;

constraint A + B + C + D <= 9;

solve satisfy;