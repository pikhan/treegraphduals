from .binary_tree import BinaryTree
import numpy as np


class GaltonWatsonTree(BinaryTree):
    """
    Galton-Watson tree from branching process.
    """

    def __init__(self, n_nodes: int = 0, offspring_dist: dict = None):
        super().__init__(n_nodes)
        self.offspring_dist = offspring_dist or {0: 0.5, 2: 0.5}

    @classmethod
    def generate(cls, max_generations: int, offspring_dist: dict):
        """
        Generate random Galton-Watson tree.

        Parameters
        ----------
        max_generations : int
            Maximum depth
        offspring_dist : dict
            {k: probability of k offspring}
        """
        nodes = []
        current_generation = [0]  # Start with root
        node_id = 1

        for gen in range(max_generations):
            next_generation = []
            for parent in current_generation:
                # Sample number of offspring
                k = np.random.choice(
                    list(offspring_dist.keys()),
                    p=list(offspring_dist.values())
                )
                for _ in range(k):
                    nodes.append((parent, node_id))
                    next_generation.append(node_id)
                    node_id += 1

            if not next_generation:
                break
            current_generation = next_generation

        # Build tree
        tree = cls(n_nodes=node_id, offspring_dist=offspring_dist)
        for parent, child in nodes:
            side = 'left' if tree.left_child[parent] == -1 else 'right'
            tree.add_edge(parent, child, side=side)

        return tree