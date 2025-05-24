% city(rome, 3).
% city(tokyo, 6).
city(bolo, 1).
city(modena, 1).

capital(X) :- 
    city(X, P), P>2.
