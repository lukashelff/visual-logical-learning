% Simple illustration of the use of lazy evaluation
%       a predicate to construct a line is called during
%       the search to obtain a linear equation
% For realistic examples a proper regression predicate will be needed
%	(usually better to implement this in C, and call it from Prolog)
% To run do the following:
%       a. Load Aleph
%       b. read_all(line).
%       c. sat(1).
%       d. reduce.

:- mode(1,p(+indep,+dep)).
:- mode(1,lin_regress1(+dep,+indep,#number,#number,#number)).

:- determination(p/2,lin_regress1/5).

:- lazy_evaluate(lin_regress1/5).

% lazy evaluation does not need substitutions from negative examples

:- positive_only(lin_regress1/5).

:- set(portray_literals,true).

aleph_portray(p(A,B)):-
	write(p(A,B)).
aleph_portray(lin_regress1(Y,X,M,C,E)):-
	write(Y), write(' is '),
	write(M), write(' * '),
	write(X), write(' + '),
	write(C), write(' +/- '),
	write(E).



nlist([]).
nlist([_|_]).

% definition for use during normal evaluation
lin_regress1(Y,X,M,C,Error):-
	number(X), number(Y),
	nonvar(M), nonvar(C), nonvar(Error), !,
	Y1 is M*X + C,
	Diff is Y - Y1,
	abs_val(Diff,D1),
	D1 =< Error.

% definition for use during lazy evaluation
lin_regress1(Y,X,M,C,Error):-
	nlist(X), nlist(Y),
	var(M), var(C), !,
	l_regress1(Y,X,M,C,Error).

% otherwise: for bottom clause
lin_regress1(Y,X,0.0,Y,0.0):-
	X \= Y.


% very simple line construction
l_regress1([YVals|_],[XVals|_],M,C,0.0):-
	YVals = [Y1,Y2|_],
	XVals = [X1,X2|_],
	M is (Y2-Y1)/(X2-X1),
	C is Y1 - M*X1.

abs_val(X,Y):- X < 0, !, Y is -X.
abs_val(X,X):- X >= 0.
