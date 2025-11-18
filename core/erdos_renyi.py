from .graph import Graph
import numpy as np


class ErdosRenyi(Graph):
    """
    Erdős–Rényi random graph G(n,p).
    """

    def __init__(self, n_nodes: int = 0, directed: bool = False):
        super().__init__(n_nodes, directed)

    @classmethod
    def generate(cls, n_nodes: int, p: float, directed: bool = False):
        """
        Generate Erdős–Rényi random graph.

        Parameters
        ----------
        n_nodes : int
            Number of nodes
        p : float
            Probability of edge between any two nodes
        directed : bool
            Whether graph is directed
        """
        graph = cls(n_nodes, directed)

        for i in range(n_nodes):
            start = i + 1 if not directed else 0
            for j in range(start, n_nodes):
                if i != j and np.random.random() < p:
                    graph.add_edge(i, j)

        return graph