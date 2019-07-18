from itertools import product

##########
# Common #
##########

class Arg:
    def __init__(self, name):
        self._name = name

def generate_functions(args):
    if isinstance(args, dict):
        for k, v in args.items():
            if isinstance(k, type) or callable(k):
                for kwargs in kwargs_generator(v):
                    def g():
                        print(kwargs)
                        return k(**kwargs)
                    yield g
            else:
                yield gen_tuples(k, v)
    elif isinstance(args, list):
        for v in args:
            yield from generate_functions(v)
    else:
        raise NotImplementedError()

def gen_tuples(k, v):
    for e in v:
        if isinstance(e, dict):
            for f in generate_functions(e):
                yield(k, f())
            # yield from gen_tuple_lists(e)
        else:
            yield (k, e)

def kwargs_generator(args):
    for param in product(*generate_functions(args)):
            yield dict(param)
