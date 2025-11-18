from .base_graph import BaseGraph
import numpy as np
from typing import List, Dict, Set
from collections import deque


class DAG(BaseGraph):
    """
    Directed Acyclic Graph.

    Uses adjacency list for general DAGs.
    Subclasses (like Tree) can override with more efficient structures.
    """

    def __init__(self, n_nodes: int = 0):
        super().__init__(n_nodes)
        self.adj_list: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        self.reverse_adj_list: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}

    def add_edge(self, u: int, v: int, **attrs):
        """Add directed edge u -> v."""
        super().add_edge(u, v, **attrs)
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if u not in self.reverse_adj_list[v]:
            self.reverse_adj_list[v].append(u)

    def get_successors(self, node: int) -> List[int]:
        """Get nodes this node points to."""
        return self.adj_list.get(node, [])

    def get_predecessors(self, node: int) -> List[int]:
        """Get nodes pointing to this node."""
        return self.reverse_adj_list.get(node, [])

    def get_neighbors(self, node: int) -> List[int]:
        """For DAGs, neighbors = successors."""
        return self.get_successors(node)

    def validate(self) -> bool:
        """Check for no cycles using DFS."""
        if self.n_nodes == 0:
            return True

        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.get_successors(node):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in range(self.n_nodes):
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    def topological_sort(self) -> List[int]:
        """Return nodes in topological order."""
        in_degree = {i: len(self.get_predecessors(i)) for i in range(self.n_nodes)}
        queue = deque([node for node in range(self.n_nodes) if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for successor in self.get_successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return result if len(result) == self.n_nodes else []