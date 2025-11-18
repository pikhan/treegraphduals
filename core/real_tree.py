from .base_graph import BaseGraph
import numpy as np


class RealTree(BaseGraph):
    """
    Real tree (R-tree): continuous metric space satisfying 0-hyperbolicity.

    This is a fundamentally different mathematical object from combinatorial trees.
    """

    def __init__(self, n_nodes: int = 0):
        super().__init__(n_nodes)
        self.metric = np.zeros((n_nodes, n_nodes))  # Distance matrix

    def validate(self) -> bool:
        """Check 0-hyperbolicity condition."""
        # For all quadruples (w,x,y,z):
        # d(w,x) + d(y,z) ≤ max(d(w,y) + d(x,z), d(x,y) + d(w,z))
        n = self.n_nodes
        if n < 4:
            return True

        # Sample check (full check is expensive)
        for _ in range(min(100, n ** 4)):
            w, x, y, z = np.random.choice(n, 4, replace=False)
            d = self.metric

            sum1 = d[w, x] + d[y, z]
            sum2 = d[w, y] + d[x, z]
            sum3 = d[x, y] + d[w, z]

            if sum1 > max(sum2, sum3) + 1e-10:  # tolerance for floating point
                return False

        return True