from collections.abc import Iterable
from itertools import chain, product
import pprint
import inspect


class Node():

    def get_name():
        raise NotImplementedError()

    def expand(self):
        raise NotImplementedError()

class FunctionNode(Node):

    def __init__(self, function, kwargs=None, name=None):
        """
        Parameters
        ----------
        function: Callable or Class
        kwargs: None or Dict[str,Node] or Dict[str,Iterable[Node]] or Iterable[Dict[str,Node]] or Iterable[Dict[str,Iterable[Node]]]; defaults to None
        name: str or None
        """
        assert isinstance(function, type) or callable(function)

        self._function = function
        self._kwargs = kwargs
        self._name = name

    def __repr__(self):
        return 'FunctionNode[f={},k={},n={}]'.format(self._function, self._kwargs, self._name)

    def get_function(self):
        return self._function

    def get_kwargs(self):
        return self._kwargs

    def get_name(self):
        return self._name

    def expand(self):
        if self._kwargs == None:
            return [FunctionNode(function=self._function, kwargs=self._kwargs, name=self._name)]
        elif isinstance(self._kwargs, dict):
            kwargs_list = self._expand_dict(self._kwargs)
            return [FunctionNode(function=self._function, kwargs=kwargs, name=self._name) for kwargs in kwargs_list]
        elif isinstance(self._kwargs, Iterable):
            kwargs_list = list(chain(*[self._expand_dict(kwargs) for kwargs in self._kwargs]))
            return [FunctionNode(function=self._function, kwargs=kwargs, name=self._name) for kwargs in kwargs_list]
        else:
            raise NotImplementedError()  # TODO: ArgumentError?

    def _expand_dict(self, d):
        args = [[(k, u) for u in v] if isinstance(v, Iterable) else [(k, v)] for k, v in d.items()]
        return [dict(combination) for combination in product(*args)]

class ValueNode(Node):

    def __init__(self, value, name=None):
        self._value = value
        self._name = name

    def __repr__(self):
        return 'ValueNode[v={},n={}]'.format(self._value, self._name)

    def get_name(self):
        return self._name

    def get_value(self):
        return self._value

    def expand(self):
        return [self]

class ValuesNode(Node):

    def __init__(self, values, name=None):
        self._values = values
        self._name = name

    def __iter__(self):
        return iter(self.expand())

    def __repr__(self):
        return 'ValuesNode[v={},n={}]'.format(self._values, self._name)

    def get_name(self):
        return self._name

    def get_values(self):
        return self._values

    def expand(self):
        return [ValueNode(v,name=self._name) for v in self._values]

class NodesNode(Node):

    def __init__(self, nodes, name):
        self._nodes = nodes
        self._name = name

    def __iter__(self):
        return iter(self.expand())

    def __repr__(self):
        return 'NodesNode[v={},n={}]'.format(self._nodes, self._name)

    def get_name(self):
        return self._name

    def get_nodes(self):
        return self._values

    def expand(self):
        results = []
        for n in self._nodes:
            if isinstance(n, FunctionNode):
                results.extend(FunctionNode(name=self._name, function=n.get_function(), kwargs=n.get_kwargs()).expand())
            elif isinstance(n, ValueNode):
                results.extend(ValueNode(name=self._name, value=n.get_value()).expand())
            elif isinstance(n, ValuesNode):
                results.extend(ValuesNode(name=self._name, values=n.get_values()).expand())
            elif isinstance(n, InjectNode):
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        return results

class InjectNode(Node):

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return 'InjectNode[n={}]'.format(self._name)

    def get_name(self):
        return self._name

    def expand(self):
        return [self]
