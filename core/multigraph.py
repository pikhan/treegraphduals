from .graph import Graph


class Multigraph(Graph):
    """
    Multigraph: allows multiple edges between same vertices.
    """

    def __init__(self, n_nodes: int = 0, directed: bool = True):
        super().__init__(n_nodes, directed)
        self.edge_list = []  # Store all edges (including duplicates)
        self.edge_multiplicities = {}

    def add_edge(self, u: int, v: int, **attrs):
        """Add edge (parallel edges allowed)."""
        edge_id = len(self.edge_list)
        self.edge_list.append((u, v, edge_id))

        key = (u, v) if self.directed else tuple(sorted([u, v]))
        self.edge_multiplicities[key] = self.edge_multiplicities.get(key, 0) + 1

        # Don't use parent add_edge (it prevents duplicates)
        self._graph.add_edge(u, v, edge_id=edge_id, **attrs)