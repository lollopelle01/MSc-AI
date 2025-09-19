var 10..40:A;
var 10..40:B;
var 10..40:C;
var 10..40:D;

constraint A in {20,40};
constraint C = B + 13;
constraint D >= C;
constraint A+B+C+D==80;

solve satisfy;