%% Dumb excercises %%
p :- write("Hello world").
p(x) :- write(x).
p(X, Output) :- Output is X+1.



%% Factorial recursion %%
factorial(0,1). %base case
factorial(PosNumber, Result) :-
    PosNumber>=0,
    Pos1 is PosNumber-1,
    factorial(Pos1, Res1),
    Result is PosNumber*Res1.



%% Define a Prolog program that receives in input a number N, and print 
%% all the numbers between 1 and N
specialPrint(1) :- write(1), nl.
specialPrint(N) :-
    N>1,
    N1 is N-1,
    specialPrint(N1),
    write(N), nl.



%% Fibonacci (??)
fibonacci(0, 0).
fibonacci(1, 1).
fibonacci(N, Res) :- 
    N>0,
    N1 is N-1, fibonacci(N1, Res1),
    N2 is N-2, fibonacci(N2, Res2),
    Res is Res1 + Res2.



%% Write a predicate about a number N, that is true if N is prime
cannotBeDividedBy(_, 1). % Base case: If D is 1, stop.
cannotBeDividedBy(N, D) :-
    D > 1,
    Rest is N mod D,
    Rest > 0,         % N is not divisible by D
    D1 is D - 1,
    cannotBeDividedBy(N, D1). % Recurse with D-1

isPrime(2). % 2 is prime
isPrime(N) :-
    D is N - 1,       % Start checking from N-1 downwards
    cannotBeDividedBy(N, D). % Check all divisors from N-1 to 1



%% Write a predicate that, given a number N, prints out all the prime 
%% numbers between 2 and N.
primePrint(N) :- isPrime(N), !, write(N), nl.
primePrint(N) :- N>0. % just empty to handle not prime case
printOnlyIfPrime(2) :- write(2), nl.
printOnlyIfPrime(N) :-
    N>2,
    N1 is N-1,
    printOnlyIfPrime(N1),
    primePrint(N).
    
