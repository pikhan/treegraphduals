from .tree import Tree
from typing import List


class Forest(Tree):
    """
    Forest: collection of disconnected trees.

    Implemented as a Tree with virtual root connecting all tree roots.
    """

    def __init__(self, trees: List[Tree] = None):
        if trees is None:
            trees = []

        # Calculate total nodes (including virtual root)
        total_nodes = 1 + sum(tree.n_nodes for tree in trees)
        super().__init__(n_nodes=total_nodes, root=0)

        self.trees = trees
        self.component_roots = []

        # Connect all tree roots to virtual root
        offset = 1
        for tree in trees:
            self.add_edge(0, offset, length=0)  # Virtual edge
            self.component_roots.append(offset)
            # Copy tree structure with offset
            for i in range(tree.n_nodes):
                if not tree.is_root(i):
                    parent = tree.parent[i] + offset
                    child = i + offset
                    self.add_edge(parent, child, length=tree.get_edge_length(tree.parent[i], i))
            offset += tree.n_nodes

    def get_connected_components(self) -> List[Tree]:
        """Get individual trees in the forest."""
        return self.trees

    def validate(self) -> bool:
        """Forest is valid if all component trees are valid."""
        return all(tree.validate() for tree in self.trees)