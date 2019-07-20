import warnings

# Tesnsorflow has all sorts of internal dependencies that are on the deprecation
# path. We're not going to fix them without upgrading tensorflow, so let's
# ignore them to reduce the noise.
warnings.filterwarnings('ignore', category=DeprecationWarning)

import rl

from rl.core.graph import FunctionNode
from rl.core.graph import GraphGenerator
from rl.core.graph import InjectNode
from rl.core.graph import NodesNode
from rl.core.graph import Specification
from rl.core.graph import ValueNode
from rl.core.graph import ValuesNode

class GraphGeneratorTest():

    # def test_me(self):
    #
    #     graph_generator = GraphGenerator(specifications=[
    #         Specification(providers=[
    #             NodesNode(name='bar', nodes=[
    #                 FunctionNode(
    #                     function=Foo,
    #                     kwargs={'name': [ValueNode(value='f')]}
    #                 )
    #             ]),
    #             FunctionNode(
    #                 name='BazNode',
    #                 function=Baz,
    #                 kwargs=[
    #                     {
    #                         'learning_rate': [ValueNode(value=3), FunctionNode(function=get_some_learning_rate)],
    #                         'foo': FunctionNode(function=Foo, kwargs={'name': ValueNode(value='foo1')}),
    #                         'bar': InjectNode(name='bar')
    #                     },
    #                     {
    #                         'learning_rate': ValueNode(value=3),
    #                         'foo': FunctionNode(function=Foo, kwargs={'name': ValueNode(value='foo2')}),
    #                         'bar': InjectNode(name='bar')
    #                     }
    #                 ]),
    #             FunctionNode(
    #                 name='FooNode',
    #                 function=Foo,
    #                 kwargs={'name': NodesNode(name='b', nodes=[ValueNode(value='afoo'), ValueNode(value='bfoo')])})
    #         ]),
    #     ])
    #
    #     for graph in graph_generator:
    #         print(graph.get('bar'), '\t', graph.get('BazNode'), '\t', graph.get('FooNode'))


    def test_value_node(self):
        value = 'value'
        value_node = ValueNode(name='value-node', value=value)
        specification = Specification(nodes=[value_node])
        graph_generator = GraphGenerator(specifications=[specification])

        assert next(graph_generator).get('value-node') == value

    def test_values_node(self):
        value_1 = 'value-1'
        value_2 = 'value-2'
        values_node = ValuesNode(name='values-node', values=[value_1, value_2])
        specification = Specification(nodes=[values_node])
        graph_generator = GraphGenerator(specifications=[specification])

        assert next(graph_generator).get('values-node') == value_1
        assert next(graph_generator).get('values-node') == value_2

    def test_function_node(self):
        function = make_a_name
        function_node = FunctionNode(name='function-node', function=function)
        specification = Specification(nodes=[function_node])
        graph_generator = GraphGenerator(specifications=[specification])

        assert next(graph_generator).get('function-node') == function()




##################
# Helper Classes #
##################

class Foo:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Foo(name={})'.format(self.name)

class Bar:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Bar(name={})'.format(self.name)

class Baz:

    def __init__(self, name, foo, bar):
        self.name = name
        self.foo = foo
        self.bar = bar

    def __repr__(self):
        return 'Baz(name={},\tfoo={},\tbar={})'.format(self.name, self.foo, self.bar)


def make_a_name():
    return '42'
