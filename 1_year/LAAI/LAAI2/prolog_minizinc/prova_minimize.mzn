int: max_colors = 6;

var 1..max_colors: A;
var 1..max_colors: B;
var 1..max_colors: C;
var 1..max_colors: D;
var 1..max_colors: E;
var 1..max_colors: F;

constraint A != D;
constraint A != B;
constraint A != C;

constraint B != D;
constraint B != E;
constraint B != C;

constraint C != E;

constraint D != E;
constraint D != F;

var int: num_colors == max([A,B,C,D,E,F]);

solve minimize num_colors;





