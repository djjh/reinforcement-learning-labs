from collections.abc import Iterable
from itertools import chain, product
import pprint
import inspect

from .nodes import *
from .graph import *


class GraphGenerator():

    def __init__(self, specifications):
        """

        Parameters
        ----------
        specifications: Iterable[Specification] -- TODO

        """
        self._specifications = specifications
        self._graph_iterator = None

        def iterator():
            for s in self._specifications:
                for g in self._build_graph_iterator(s):
                    yield g

        self._iterator = iterator()

    def __iter__(self):
        return self._iterator

    def __next__(self):
        return next(self._iterator)

    def _build_graph_iterator(self, specification):

        # 1. Expand specification combinations into multiple flat specifications.
        specifications = self._expand(specification)
        # [print(s.get_providers(), '\n') for s in specifications]

        for specification in specifications:
            # 2. Expand into node lists.
            specification._build_nodes()
            specification._build_edges()

            # 3. Sort nodes topologically
            specification._sort_topologically()

        def generator():
            for specification in specifications:
                # 4. Instanitate nodes, injecting dependencies, and set graph...
                specification._complete()
                yield Graph(specification.top)

        return generator()


    def _expand(self, specification):
        """
        Returns list of expanded specifications.
        """
        expanded_nodes = [node.expand() for node in specification.get_providers()]
        return [Specification(nodes=nodes) for nodes in product(*expanded_nodes)]
