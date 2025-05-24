%% Parent-Child relations %%
parent(charles, william).
parent(charles, harry).
parent(diana, william).
parent(diana, harry).

parent(elizabeth, charles).
parent(elizabeth, anne).
parent(elizabeth, andrew).
parent(elizabeth, edward).

parent(philip, charles).
parent(philip, anne).
parent(philip, andrew).
parent(philip, edward).

parent(anne, peter).
parent(anne, zara).
parent(mark, peter).
parent(mark, zara).

parent(andrew, beatrice).
parent(andrew, eugenie).
parent(sarah, beatrice).
parent(sarah, eugenie).

parent(edward, louise).
parent(edward, james).
parent(sophie, louise).
parent(sophie, james).

parent(spencer, diana).
parent(kydd, diana).

%% Marriage relations %%
spouse(charles, diana).
spouse(elizabeth, philip).
spouse(anne, mark).
spouse(andrew, sarah).
spouse(edward, sophie).
spouse(spencer, kydd).

%% Child-Parent relation %%
child(X, Y) :- parent(Y, X).    % it is just the reverse of parent-child

%% Sibilig relations %%
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

%% Father and Mother differentiation %%
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

%% Gender definitions %%
male(charles).
male(william).
male(harry).
male(philip).
male(andrew).
male(edward).
male(mark).
male(peter).
male(james).
male(spencer).

female(diana).
female(elizabeth).
female(anne).
female(zara).
female(beatrice).
female(eugenie).
female(sarah).
female(sophie).
female(louise).
female(kydd).

% NOTE: you can express the nephews of charles by asking
% `parent(charles, Y), child(X, Y).`