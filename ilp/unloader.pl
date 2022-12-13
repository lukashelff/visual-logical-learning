unload_last_source:-
  findall(Source, source_file(Source), LSource),
  reverse(LSource, [Source|_]),
  unload_source(Source).

unload_source(Source):-
  ground(Source),
  source_file(Pred, Source),
  functor(Pred, Functor, Arity),
  abolish(Functor/Arity),
  fail.

unload_source(_).