from collections.abc import Iterable
from itertools import chain, product
import pprint
import inspect

from .nodes import *

class Graph():

    def __init__(self, graph):
        self._graph = graph

    def get(self, key):
        return self._graph[key]

class Specification():

    def __init__(self, nodes):
        self._providers = nodes

    def get_providers(self):
        return self._providers

    def _build_nodes(self):
        """ NOTE: only works if flattened! """
        stack = list(self._providers)
        nodes = set()
        while len(stack) > 0:
            node = stack.pop()
            nodes.add(node)
            if isinstance(node, ValueNode):
                pass
            elif isinstance(node, ValuesNode):
                print(node)
                raise Error("This shouldn't happen.")
            elif isinstance(node, FunctionNode):
                kwargs = node.get_kwargs()
                if isinstance(kwargs, dict):
                    for k, v in kwargs.items():
                        if not isinstance(v, InjectNode):
                            stack.append(v)
                elif isinstance(kwargs, Iterable):
                    for d in kwargs:
                        for k, v in d.items():
                            if not isinstance(v, InjectNode):
                                stack.append(v)
            else:
                raise NotImplementedError("{} not implemented for Specification._build_nodes() ".format(type(node)))

        self.nodes = nodes
        self.nodes_by_name = dict([(n.get_name(),n) for n in self.nodes])
        if None in self.nodes_by_name:
            del self.nodes_by_name[None]

    def _build_edges(self):
        """ NOTE: only works if flattened! """
        stack = list(self._providers)
        edges = {}
        resolved = {}
        while len(stack) > 0:
            node = stack.pop()
            resolved[node] = {}
            if isinstance(node, FunctionNode):
                kwargs = node.get_kwargs()
                if isinstance(kwargs, dict):
                    for k, v in kwargs.items():
                        stack.append(v)
                        edges[node] = edges.get(node, [])
                        if isinstance(v, InjectNode):
                            resolved[node][v] = self.nodes_by_name[v.get_name()]
                            edges[node].append(self.nodes_by_name[v.get_name()])
                        else:
                            resolved[node][v] = v
                            edges[node].append(v)
                elif isinstance(kwargs, Iterable):
                    for d in kwargs:
                        for k, v in d.items():
                            stack.append(v)
                            edges[node] = edges.get(node, [])
                            if isinstance(v, InjectNode):
                                resolved[node][v] = self.nodes_by_name[v.get_name()]
                                edges[node].append(self.nodes_by_name[v.get_name()])
                            else:
                                resolved[node][v] = v
                                edges[node].append(v)

        # [print(e, '\n') for e in edges]
        self.resolved = resolved
        self.edges = edges

    def _sort_topologically(self):
        """
        Kahn's algorithm for Topological Sorting

        L ← Empty list that will contain the sorted elements
        S ← Set of all nodes with no incoming edge
        while S is non-empty do
            remove a node n from S
            add n to tail of L
            for each node m with an edge e from n to m do
                remove edge e from the graph
                if m has no other incoming edges then
                    insert m into S
        if graph has edges then
            return error   (graph has at least one cycle)
        else
            return L   (a topologically sorted order)
        """
        N = set(self.nodes)
        E = dict(self.edges)
        L = []
        S = get_nodes_indegree_zero(N, E)
        while len(S) > 0:
            n = S.pop()
            L.append(n)
            for m in get_outs(n, E):
                E[m].remove(n)
                if get_indegree(m, E) == 0:
                    S.add(m)

        remaining = [y for x in E.values() for y in x]
        if len(remaining) > 0:
            print('\nERRROR')
            print(remaining)

        self._topologically_sorted_nodes = L

    def _complete(self):
        instances = {}

        # Instaniate, saving map from node to instance.
        for node in self._topologically_sorted_nodes:
            if isinstance(node, ValueNode):
                instances[node] = node._value
            elif isinstance(node, ValuesNode):
                raise Error("This shouldn't happen.")
            elif isinstance(node, FunctionNode):
                if node.get_kwargs() == None:
                     instances[node] = node.get_function()()
                elif isinstance(node.get_kwargs(), dict):
                    kwargs = {}
                    for k, v in node.get_kwargs().items():
                        print(k, v)
                        rv = self.resolved[node][v]
                        print(rv)
                        kwargs[k] = instances[rv]
                    # print(inspect.getargspec(node._function))
                    # for a in missingArgs(node._function, kwargs):
                    #     print(a)
                    # print(kwargs.keys())
                    # print(dir(node._function))
                    instances[node] = node.get_function()(**kwargs)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError("{} not implemented for Specification._complete() ".format(type(node)))

        # Generate a map from top level object (name, class) or (name, function) pairs to instance.
        top = {}
        for n in self.get_providers():
            top[n.get_name()] = instances[n]
        if None in top:
            del top[None]

        self.instances = instances
        self.top = top

def get_nodes_indegree_zero(nodes, edges):
    res = set()
    for node in nodes:
        indeg = get_indegree(node, edges)
        if indeg == 0:
            res.add(node)
    return res

def get_indegree(node, edges):
    return len(get_ins(node, edges))

def get_ins(node, edges):
    ins = set()
    for k, v in edges.items():
        if node == k:
            ins.update(v)
    return ins

def get_outs(node, edges):
    outs = set()
    for k, v in edges.items():
        if node in v:
            outs.add(k)
    return outs

def getRequiredArgs(func):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    return args   # *args and **kwargs are not required, so ignore them.

def missingArgs(func, argdict):
    return set(getRequiredArgs(func)).difference(argdict)
