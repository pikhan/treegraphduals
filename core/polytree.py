from .tree import Tree
from .dag import DAG


class Polytree(Tree):
    """
    Polytree: DAG whose undirected version is a tree.

    Nodes can have multiple parents (unlike regular trees).
    """

    def __init__(self, n_nodes: int = 0):
        # Note: Don't use Tree's parent array (nodes can have multiple parents)
        DAG.__init__(self, n_nodes)
        self.parents = {i: [] for i in range(n_nodes)}
        self.children = [[] for _ in range(n_nodes)]

    def add_edge(self, u: int, v: int, **attrs):
        """Add directed edge u -> v."""
        DAG.add_edge(self, u, v, **attrs)
        self.parents[v].append(u)
        if v not in self.children[u]:
            self.children[u].append(v)

    def validate(self) -> bool:
        """
        Validate polytree structure.

        Must be:
        - A DAG
        - Undirected version forms a tree (no cycles when ignoring direction)
        """
        if not DAG.validate(self):
            return False

        # Check undirected version is a tree
        # Convert to undirected and check for cycles
        visited = set()

        def has_undirected_cycle(node, parent):
            visited.add(node)
            # Check all neighbors (both children and parents)
            neighbors = set(self.children[node]) | set(self.parents[node])
            for neighbor in neighbors:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True
                if has_undirected_cycle(neighbor, node):
                    return True
            return False

        # Check from any node
        if self.n_nodes > 0:
            return not has_undirected_cycle(0, -1)
        return True