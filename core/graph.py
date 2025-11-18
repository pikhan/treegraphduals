from .base_graph import BaseGraph


class Graph(BaseGraph):
    """
    General graph with no constraints.

    Allows cycles, multiple components, etc.
    """

    def __init__(self, n_nodes: int = 0, directed: bool = True):
        super().__init__(n_nodes)
        self.directed = directed
        self.adj_list = {i: [] for i in range(n_nodes)}

    def add_edge(self, u: int, v: int, **attrs):
        """Add edge."""
        super().add_edge(u, v, **attrs)
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if not self.directed and u not in self.adj_list[v]:
            self.adj_list[v].append(u)

    def validate(self) -> bool:
        """General graphs are always valid."""
        return True