var 0..1: A;
var 0..1: B;
var 0..1: C;
var 0..1: D;

var int: W;
var int: V;

% Aw = A*10; Av = A*500;
% Bw = B*4; Bv = B*600;
% Cw = C*2; Cv = C*100;
% Dw = D*2; Dv = D*200;

W = A*10 + B*4 + C*2 + D*2;
V = A*500 + B*600 + C*100 + D*200;

constraint  W <= 11;
constraint  V > 600;

solve satisfy;