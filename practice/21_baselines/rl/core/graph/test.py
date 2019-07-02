from nodes import *
from graph import *
from graph_generator import *

def test():

    class Foo:
        def __init__(self, name):
            self._name = name

        def __repr__(self):
            return 'Foo(name={})'.format(self._name)

    class Bar:
        def __init__(self, name):
            self._name = name

        def __repr__(self):
            return 'Bar(name={})'.format(self._name)

    class Baz:
        def __init__(self, learning_rate, foo, bar):
            self._learning_rate = learning_rate
            self._foo = foo
            self._bar = bar

        def __repr__(self):
            return 'Baz(learning_rate={},\tfoo={},\tbar={})'.format(self._learning_rate, self._foo, self._bar)


    def get_some_learning_rate():
        return 42

    graph_generator = GraphGenerator(specifications=[
        Specification(providers=[
            ValuesNode(
                name='bar3',
                values=['zzz', 'bbb']),
            FunctionNode(
                name='bar',
                function=Bar,
                kwargs={'name': [ValueNode(value='bar1'), ValueNode(value='bar2'), InjectNode(name='bar3')]}),
            FunctionNode(
                name='BazNode',  # name: str or None; defaults to None
                function=Baz,    # function: Callable or Class; Required
                kwargs=[
                    {
                        'learning_rate': [ValueNode(value=3), FunctionNode(function=get_some_learning_rate)],
                        'foo': FunctionNode(function=Foo, kwargs={'name': ValueNode(value='foo1')}),
                        'bar': InjectNode(name='bar')
                    },
                    {
                        'learning_rate': ValueNode(value=3),
                        'foo': FunctionNode(function=Foo, kwargs={'name': ValueNode(value='foo2')}),
                        'bar': InjectNode(name='bar')
                    }
                ]),
            FunctionNode(
                name='FooNode',
                function=Foo,
                kwargs={'name': [ValueNode(value='foofoo1'), ValueNode(value='foofoo2')]})
        ]),
    ])

    for graph in graph_generator:
        print(graph.get((Baz, 'BazNode')), '\t', graph.get((Foo, 'FooNode')))


test()
