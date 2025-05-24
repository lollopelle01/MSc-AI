% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%% Vanilla %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % 1) Base case of the empty goal
% % solve(true) :- !.

% % % 2) Conjunctions of sub-goals
% % solve((A,B)) :- 
% %     !, % to be sure not to match other rules
% %     solve(A), solve(B). % left -> right (left most)
% % % 3) Single goal
% % solve(A) :- 
% %     clause(A,B), % for the query A extracts the body B
% %     solve(B). % solves the body




% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%% Vanilla rigth-most %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % solve(true) :- !.
% % solve((A,B)) :- !, solve(B), solve(A). % right -> left (right most)
% % solve(A) :- clause(A,B), solve(B). 



% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%% Other interpreters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % Define a Prolog interpreter solve(Goal, Step)that:
% % % • It is true if Goal can be proved
% % % • In case Goal is proved, Step is the number of resolution steps used to prove the goal
% % % – In case of conjunctions, the number of steps is defined as the sum of the steps 
% % % needed for each atomic conjunct
% % solve(true, 0) :- !.
% % solve((A,B), Step) :- 
% %     !, 
% %     solve(A, StepA), 
% %     solve(B, StepB),
% %     Step is StepA + StepB.
% % solve(A, Step) :-
% %     clause(A,B), 
% %     solve(B, StepB),
% %     Step is StepB + 1.


% %% Test problem %
% a :- b, c.      %      
% b :- d.         %      
% c.              %
% d.              %
% %%%%%%%%%%%%%%%%%

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % Define a meta-interpreter solve(Goal,CF), that is true if Goalcan be proved, with certainty CF.
% % % • For conjunctions, the certainty is the minimum of the certainties of the conjuncts
% % % • For rules, the certainty is the product of the certainty of the rule itself times the 
% % % certainty of the proof of the body (eventually divided by 100).

% % min(A,B,A) :- A < B, !.
% % min(_,B,B).

% % solve(true, 100) :- !.
% % solve((A,B), CFmin) :-
% %     !,
% %     solve(A, CFa),
% %     solve(B, CFb),
% %     min(CFa, CFb, CFmin).
% % solve(A, CFa) :-
% %     rule(A, B, CF),  %  extract the body B of the rule and its certainty
% %     solve(B, CFb),
% %     CFa is ((CFb*CF)/100).



% % %% Test problem
% % rule(a, (b,c), 10).
% % rule(a, d, 90).
% % rule(b,true, 100).
% % rule(c,true, 50).
% % rule(d,true, 100).
% % %%

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % Interactive interpreter that handles unknowns by asking the used

% % % • Idea: extend your KB and/or reasoning tool, so that:
% % % – It tries to prove a Goal using the given KB
% % % – If it fails, the reasoner could also ask help to the user

% % % NOTE: in this way we check all the falses and not only the falses for missing predicate

% % solve(true) :- !.
% % solve((A,B)) :-
% %     !,
% %     solve(A),
% %     solve(B).
% % solve(A) :-
% %     clause(A,B),
% %     solve(B).
% % solve(A) :- % if previous failed
% %     askable(A), % A is not a predicte not to ask about
% %     format("Is ~w true? (yes/no)~n", [A]),
% %     read(Answer),
% %     Answer == yes.


% % %% Test problem
% % % Askables
% % askable(tweets(_)).
% % askable(small(_)).
% % askable(cuddly(_)).
% % askable(has_feathers(_)).
% % % KB
% % good_pet(X) :- bird(X), small(X).
% % good_pet(X) :- cuddly(X), yellow(X).
% % bird(X) :- has_feathers(X), tweets(X).
% % yellow(tweety).
% % %%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Write a meta-interpreter for the Prolog language that prints out, before 
% % and after the execution of a subgoal, the subgoal itself. 
 
% solve(X) :- solve(X, 0). % interface

% solve(true, _) :- !.
% solve((A,B), N) :- 
%     !,
%     solve(A, N),
%     solve(B, N).
% solve(A, N) :-
%     tt(N), format("Solving: ~w~n", [A]),
%     clause(A,B),
%     N1 is N + 1,
%     tt(N1), format("Extracted rule: ~w :- ~w~n", [A,B]),
%     solve(B, N1),
%     tt(N1), format("Solved: ~w~n", [B]).

% % tab function
% tt(0) :- !.
% tt(N) :-
%     N > 0,
%     tab(3), % tab unit
%     N1 is N - 1,
%     tt(N1).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
