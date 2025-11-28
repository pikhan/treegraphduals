"""
Tree data structure with multi-library compatibility.

Provides efficient tree representation using parent pointers and children lists,
with conversions to NetworkX, igraph, and other formats.
"""

import numpy as np
from typing import Optional, List, Dict, Set, Tuple, Any, Union
from collections import deque, defaultdict

from .base_graph import BaseGraph


class Tree(BaseGraph):
    """
    General tree structure with efficient parent-child representation.
    
    Internal representation uses:
    - Parent array: O(1) parent lookup
    - Children lists: O(1) child access
    - Edge attributes: stored separately
    
    Compatible with NetworkX, igraph, and numpy/scipy.
    """
    
    def __init__(self, n_nodes: int = 0, root: Optional[int] = None):
        """
        Initialize a tree
        """
        super().__init__(n_nodes)
        
        # Tree-specific data structures
        self.root = root if root is not None else 0
        self.parent: np.ndarray = np.full(n_nodes, -1, dtype=np.int32)  # -1 means no parent
        self.children: List[List[int]] = [[] for _ in range(n_nodes)]
        self.edge_lengths: Dict[Tuple[int, int], float] = {}
        
        # Tree metrics (computed on demand)
        self._depth_cache: Optional[np.ndarray] = None
        self._subtree_sizes_cache: Optional[np.ndarray] = None
        self._dfs_order_cache: Optional[List[int]] = None
    
    def add_edge(self, parent: int, child: int, length: float = 1.0, **attrs):
        """
        Add an edge from parent to child.
        
        Parameters
        ----------
        parent : int
            Parent node index
        child : int
            Child node index
        length : float
            Edge length/weight
        **attrs : dict
            Additional edge attributes
        """
        # Update parent-child structures
        self.parent[child] = parent
        if child not in self.children[parent]:
            self.children[parent].append(child)
        
        # Store edge length
        self.edge_lengths[(parent, child)] = length
        
        # Update underlying graph representation
        super().add_edge(parent, child, **{'length': length, **attrs})
        
        # Invalidate caches
        self._invalidate_tree_caches()
    
    def set_parent(self, child: int, parent: int, length: float = 1.0):
        """Set the parent of a node (alternative to add_edge)."""
        self.add_edge(parent, child, length=length)
    
    def _invalidate_tree_caches(self):
        """Invalidate cached tree metrics."""
        self._depth_cache = None
        self._subtree_sizes_cache = None
        self._dfs_order_cache = None
    
    def get_children(self, node: int) -> List[int]:
        """
        Get list of children for a node.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_children(2)
        [4, 5]
        >>> tree.get_children(1)
        [3]
        >>> tree.get_children(6)
        []
        """
        return self.children[node]
    
    def get_parent(self, node: int) -> int:
        """
        Get parent of a node. Returns -1 if node is root.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_parent(0)
        -1
        >>> tree.get_parent(6)
        3
        >>> tree.get_parent(3)
        1
        """
        return int(self.parent[node])
    
    def is_leaf(self, node: int) -> bool:
        """
        Check if node is a leaf.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.is_leaf(2)
        False
        >>> tree.is_leaf(0)
        False
        >>> tree.is_leaf(4)
        True
        >>> tree.is_leaf(6)
        True
        """
        return len(self.children[node]) == 0
    
    def is_root(self, node: int) -> bool:
        """
        Check if node is the root.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.is_root(0)
        True
        >>> tree.is_root(1)
        False
        >>> tree.is_root(6)
        False
        >>> tree.is_root(4)
        False
        """
        return bool(self.parent[node] == -1)
    
    def get_leaves(self) -> List[int]:
        """
        Get all leaf nodes.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_leaves()
        [4, 5, 6]
        """
        return [i for i in range(self.n_nodes) if self.is_leaf(i)]
    
    def get_internal_nodes(self) -> List[int]:
        """
        Get all internal (non-leaf) nodes.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_internal_nodes()
        [1, 2, 3]
        """
        return [i for i in range(self.n_nodes) if not self.is_leaf(i) and not self.is_root(i)]
    
    def depth_first_search(self, start: Optional[int] = None) -> List[int]:
        """
        Perform depth-first search traversal.
        
        Returns nodes in DFS order (leftmost to rightmost leaves).

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.depth_first_search()
        [0, 1, 3, 6, 2, 4, 5]
        """
        if self._dfs_order_cache is not None:
            return self._dfs_order_cache.copy()
        
        if start is None:
            start = self.root
        
        order = []
        stack = [start]
        
        while stack:
            node = stack.pop()
            order.append(node)
            # Add children in reverse order so leftmost is processed first
            for child in reversed(self.children[node]):
                stack.append(child)
        
        if start == self.root:
            self._dfs_order_cache = order
        
        return order
    
    def breadth_first_search(self, start: Optional[int] = None) -> List[int]:
        """
        Perform breadth-first search traversal.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.breadth_first_search()
        [0, 1, 2, 3, 4, 5, 6]
        """
        if start is None:
            start = self.root
        
        order = []
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            order.append(node)
            queue.extend(self.children[node])
        
        return order
    
    def get_depth(self, node: Optional[int] = None) -> Union[int, np.ndarray]:
        """
        Get depth of a node or all nodes.
        
        Parameters
        ----------
        node : int, optional
            If provided, return depth of this node.
            If None, return array of all node depths.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_depth(0)
        0
        >>> tree.get_depth(6)
        3
        >>> [int(x) for x in tree.get_depth()]
        [0, 1, 1, 2, 2, 2, 3]
        """
        if self._depth_cache is None:
            depths = np.zeros(self.n_nodes, dtype=np.int32)
            for n in self.breadth_first_search():
                if not self.is_root(n):
                    depths[n] = depths[self.parent[n]] + 1
            self._depth_cache = depths
        
        if node is not None:
            return int(self._depth_cache[node])
        return self._depth_cache.copy()
    
    def get_subtree_size(self, node: Optional[int] = None) -> Union[int, np.ndarray]:
        """
        Get size of subtree rooted at node (including node itself).
        
        Parameters
        ----------
        node : int, optional
            If provided, return subtree size for this node.
            If None, return array of all subtree sizes.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_subtree_size(1)
        3
        >>> [int(x) for x in tree.get_subtree_size()]
        [7, 3, 3, 2, 1, 1, 1]
        """
        if self._subtree_sizes_cache is None:
            sizes = np.ones(self.n_nodes, dtype=np.int32)
            # Process in reverse DFS order (bottom-up)
            for n in reversed(self.depth_first_search()):
                for child in self.children[n]:
                    sizes[n] += sizes[child]
            self._subtree_sizes_cache = sizes
        
        if node is not None:
            return int(self._subtree_sizes_cache[node])
        return self._subtree_sizes_cache.copy()
    
    def get_path_to_root(self, node: int) -> List[int]:
        """
        Get path from node to root.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_path_to_root(4)
        [4, 2, 0]
        """
        path = []
        current = node
        while current != -1:
            path.append(int(current))
            current = int(self.parent[current])
        return path
    
    def get_path_between(self, node1: int, node2: int) -> List[int]:
        """
        Get path between two nodes.
        
        Returns the unique path in the tree from node1 to node2.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_path_between(4, 1)
        [4, 2, 0, 1]
        """
        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)
        
        # Find lowest common ancestor
        path1_set = set(path1)
        lca = None
        for node in path2:
            if node in path1_set:
                lca = node
                break
        
        # Build path
        path_to_lca = []
        current = node1
        while current != lca:
            path_to_lca.append(int(current))
            current = self.parent[current]
        path_to_lca.append(lca)
        
        path_from_lca = []
        current = node2
        while current != lca:
            path_from_lca.append(int(current))
            current = self.parent[current]
        
        return path_to_lca + list(reversed(path_from_lca))
    
    def get_distance(self, node1: int, node2: int, weighted: bool = True) -> float:
        """
        Get distance between two nodes.
        
        Parameters
        ----------
        node1, node2 : int
            Node indices
        weighted : bool
            If True, use edge lengths. If False, count edges.

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree.get_distance(4, 1)
        3.5
        """
        path = self.get_path_between(node1, node2)
        
        if not weighted:
            return len(path) - 1
        
        distance = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Check both directions since path might go up or down
            edge = (u, v) if (u, v) in self.edge_lengths else (v, u)
            distance += self.edge_lengths.get(edge, 1.0)
        
        return distance
    
    def traverse_edges(self, mode: str = 'depth_first') -> List[Tuple[int, int, str]]:
        """
        Traverse edges of the tree.
        
        Parameters
        ----------
        mode : str
            'depth_first': DFS traversal
            'contour': Tree contour (edges traversed twice)
        
        Returns
        -------
        List of (parent, child, direction) tuples where direction is 'down' or 'up'

        Examples
        --------
        Build tree structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(2, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> [(int(start), int(stop), direction) for (start, stop, direction) in tree.traverse_edges()]
        [(0, 1, 'down'), (1, 3, 'down'), (3, 6, 'down'), (0, 2, 'down'), (2, 4, 'down'), (2, 5, 'down')]
        >>> [(int(start), int(stop), direction) for (start, stop, direction) in tree.traverse_edges(mode='contour')]
        [(0, 1, 'down'), (1, 3, 'down'), (3, 6, 'down'), (6, 3, 'up'), (3, 1, 'up'), (1, 0, 'up'), (0, 2, 'down'), (2, 4, 'down'), (4, 2, 'up'), (2, 5, 'down'), (5, 2, 'up'), (2, 0, 'up')]
        """
        if mode == 'depth_first':
            edges = []
            for node in self.depth_first_search():
                if not self.is_root(node):
                    edges.append((self.parent[node], node, 'down'))
            return edges
        
        elif mode == 'contour':
            # Tree contour: traverse each edge twice (down and up), the tree traversal function in Zoe Haskell's thesis
            edges = []
            
            def contour_dfs(node):
                for child in self.children[node]:
                    edges.append((node, child, 'down'))
                    contour_dfs(child)
                    edges.append((child, node, 'up'))
            
            contour_dfs(self.root)
            return edges
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def horton_strahler_order(self) -> Dict[int, int]:
        """
        Compute Horton-Strahler order for all nodes.

        The Horton-Strahler order is defined recursively:
        - Leaves have order 1
        - If a node has children with maximum order k, and exactly one child
          has order k, then the node has order k
        - If a node has children with maximum order k, and two or more children
          have order k, then the node has order k+1

        Works for general trees (not just binary), planted or unplanted.

        Returns
        -------
        Dict[int, int]
            Mapping from node to Horton-Strahler order

        Examples
        --------
        Binary tree:
              0
             / \\
            1   2
           / \\   \\
          3   4   5

        >>> tree = Tree(n_nodes=6, root=0)
        >>> tree.add_edge(0, 1)
        >>> tree.add_edge(0, 2)
        >>> tree.add_edge(1, 3)
        >>> tree.add_edge(1, 4)
        >>> tree.add_edge(2, 5)
        >>> orders = tree.horton_strahler_order()
        >>> orders[1] # Internal vertex
        2
        >>> orders[0]  # Root
        2
        >>> orders[3]  # Leaf
        1

        Non-binary tree:
              0
            / | \
           1  2  3
          /|
         4 5

        >>> tree2 = Tree(n_nodes=6, root=0)
        >>> tree2.add_edge(0, 1)
        >>> tree2.add_edge(0, 2)
        >>> tree2.add_edge(0, 3)
        >>> tree2.add_edge(1, 4)
        >>> tree2.add_edge(1, 5)
        >>> orders2 = tree2.horton_strahler_order()
        >>> orders2[0]
        2
        >>> orders2 = [value for key, value in sorted(tree2.horton_strahler_order().items())]
        >>> orders2
        [2, 2, 1, 1, 1, 1]

        Another binary tree example:
             0
            /\
           1  2
          /\\  /\
         3 4  5 6
           /\\
          7  8
        >>> tree3 = Tree.from_parent_array([-1, 0, 0, 1, 1, 2, 2, 4, 4], root=0)
        >>> orders3 = [value for key, value in sorted(tree3.horton_strahler_order().items())]
        >>> orders3
        [3, 2, 2, 1, 2, 1, 1, 1, 1]
        """
        orders = {}

        # Process nodes in post-order (children before parents)
        def compute_order(node):
            children = self.get_children(node)

            # Base case: leaves have order 1
            if not children:
                orders[node] = 1
                return 1

            # Recursively compute orders for all children
            child_orders = [compute_order(child) for child in children]

            # Find maximum order among children
            max_order = max(child_orders)

            # Count how many children have the maximum order
            count_max = sum(1 for order in child_orders if order == max_order)

            # Horton-Strahler rule:
            # - If only one child has max order: keep that order
            # - If two or more children have max order: increment
            if count_max == 1:
                orders[node] = max_order
            else:
                orders[node] = max_order + 1

            return orders[node]

        compute_order(self.root)
        return orders

    def is_planted(self) -> bool:
        """
        Checks if a tree is planted (root has degree 1) or is otherwise stemless, this impacts how the overall Horton-Strahler order of the tree corresponds to the number of successive Horton prunings required to eliminate a tree
        >>> tree = Tree.from_parent_array([-1, 0, 0, 1, 1, 2, 2, 4, 4], root=0)
        >>> tree.is_planted()
        False
        >>> tree2 = Tree.from_parent_array([-1, 0, 1, 1])
        >>> tree2.is_planted()
        True
        """
        if len(self.get_children(self.root)) == 1:
            return True
        return False

    def horton_strahler_order_tree(self) -> int:
            return max(self.horton_strahler_order().values())

    def max_horton_prunings(self) -> int:
        if self.is_planted():
            return self.horton_strahler_order_tree() + 1
        return self.horton_strahler_order_tree()

    def horton_prune(self) -> 'Tree':
        """
        Perform Horton pruning on the tree.

        Horton pruning removes all branches that are not part of the main stem.
        The main stem is the path from root to a leaf that follows the highest
        Horton-Strahler orders at each node.

        At each internal node, we keep only the child with the highest order.
        If multiple children share the highest order, we pick the leftmost one.

        Also performs series reduction: removes degree-2 nodes (nodes with
        exactly one child) by connecting their parent directly to their child.

        Works for general trees (not just binary), planted or unplanted.

        Returns
        -------
        Tree
            New pruned tree (original tree is not modified)

        Examples
        --------
              0             0            0
             /\\            /\\         /\\
            1   2         1  2         1  2
           /\\   /  ->    /\\    ->
          3 4  5        3
         /
        6
        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(1, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> pruned = tree.horton_prune()
        >>> pruned.n_nodes
        3
        >>> tree2 = Tree.from_parent_array([-1, 0, 0, 1, 1, 2, 2, 4, 4], root=0)
        >>> tree2.max_horton_prunings()
        3
        >>> tree2 = tree2.horton_prune()
        >>> tree2.n_nodes
        3
        >>> tree2 = tree2.horton_prune()
        >>> tree2.n_nodes
        1
        >>> tree2 = tree2.horton_prune()
        >>> tree2.n_nodes
        0
        >>> tree2
        Tree(n_nodes=0, n_leaves=0, root=-1)
        """
        # Find all non-leaf nodes
        internal_nodes = [i for i in range(self.n_nodes) if not self.is_leaf(i)]

        # Edge case: no internal nodes means tree is empty or just leaves
        if len(internal_nodes) == 0:
            return Tree(n_nodes=0, root=-1)  # Tree has been fully pruned, a -1 root should indicate an empty (pointless) tree

        # Edge case: only root remains (single node)
        if len(internal_nodes) == 1 and internal_nodes[0] == self.root:
            return Tree(n_nodes=1, root=0)

        # Map old to new IDs
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(internal_nodes)}

        # Create new tree
        pruned = Tree(n_nodes=len(internal_nodes), root=old_to_new[self.root])

        # Add edges between internal nodes
        for old_node in internal_nodes:
            if old_node != self.root:
                old_parent = int(self.parent[old_node])
                # Parent must also be internal (otherwise disconnected)
                if old_parent in internal_nodes:
                    new_parent = old_to_new[old_parent]
                    new_child = old_to_new[old_node]
                    edge_length = self.get_edge_length(old_parent, old_node)
                    pruned.add_edge(new_parent, new_child, length=edge_length)

        # Series reduction
        return pruned._series_reduction()

    def _series_reduction(self) -> 'Tree':
        """
        Remove degree-2 nodes (series reduction).

        A degree-2 node has exactly one parent and one child.
        We remove it by connecting its parent directly to its child,
        with edge length = sum of the two removed edges.

        Returns
        -------
        Tree
            New tree with degree-2 nodes removed

        >>> tree = Tree(n_nodes=7, root=0)
        >>> tree.add_edge(0, 1, length=1.0)
        >>> tree.add_edge(0, 2, length=1.5)
        >>> tree.add_edge(1, 3, length=2.0)
        >>> tree.add_edge(1, 4, length=1.0)
        >>> tree.add_edge(2, 5, length=1.0)
        >>> tree.add_edge(3, 6, length=0.5)
        >>> tree = tree._series_reduction()
        >>> tree.n_nodes
        5
        """
        # Find degree-2 nodes (not root, exactly 1 child)
        degree_2_nodes = set()
        for node in range(self.n_nodes):
            if not self.is_root(node) and len(self.children[node]) == 1:
                degree_2_nodes.add(node)

        if not degree_2_nodes:
            return self  # No reduction needed

        # Build new tree without degree-2 nodes
        nodes_to_keep = [i for i in range(self.n_nodes) if i not in degree_2_nodes]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(nodes_to_keep)}

        reduced = Tree(n_nodes=len(nodes_to_keep), root=old_to_new[self.root])

        # Add edges, skipping over degree-2 nodes
        for old_node in nodes_to_keep:
            if old_node != self.root:
                # Trace back to first non-degree-2 ancestor
                ancestor = int(self.parent[old_node])
                total_length = self.get_edge_length(ancestor, old_node)

                while ancestor in degree_2_nodes:
                    grandparent = int(self.parent[ancestor])
                    total_length += self.get_edge_length(grandparent, ancestor)
                    ancestor = grandparent

                # Add edge from ancestor to current node
                new_parent = old_to_new[ancestor]
                new_child = old_to_new[old_node]
                reduced.add_edge(new_parent, new_child, length=total_length)

        return reduced
    
    def validate(self) -> bool:
        """
        Validate tree structure.
        
        Checks:
        - Exactly one root (node with no parent)
        - All other nodes have exactly one parent
        - No cycles
        - Connected
        """
        # Check single root
        roots = [i for i in range(self.n_nodes) if self.parent[i] == -1]
        if len(roots) != 1:
            return False
        
        # Check all nodes reachable from root (connected)
        reachable = set(self.breadth_first_search())
        if len(reachable) != self.n_nodes:
            return False
        
        # Check no cycles (DFS with parent tracking)
        visited = set()
        
        def has_cycle(node, parent_node):
            visited.add(node)
            for child in self.children[node]:
                if child in visited:
                    return True
                if has_cycle(child, node):
                    return True
            return False
        
        if has_cycle(self.root, -1):
            return False
        
        return True

    @classmethod
    def from_adjacency_matrix(cls, adj: np.ndarray, weighted: bool = False,
                              root: Optional[int] = None) -> 'Tree':
        """
        Create tree from adjacency matrix.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix (n x n)
        weighted : bool
            If True, matrix values are edge weights
        root : int, optional
            Root node. If None, inferred as node with no incoming edges.

        Returns
        -------
        Tree
            Reconstructed tree

        Examples
        --------
        >>> tree = Tree(n_nodes=3, root=0)
        >>> tree.add_edge(0, 1, length=2.0)
        >>> tree.add_edge(0, 2, length=3.0)
        >>> adj = tree.to_adjacency_matrix(weighted=True, weight_attr='length')
        >>> tree2 = Tree.from_adjacency_matrix(adj, weighted=True)
        >>> tree2.n_nodes
        3
        >>> tree2.get_edge_length(0, 1)
        2.0
        """
        n_nodes = adj.shape[0]

        # Find root (node with no incoming edges)
        if root is None:
            in_degree = np.sum(adj > 0, axis=0)  # Count incoming edges
            root_candidates = np.where(in_degree == 0)[0]
            if len(root_candidates) == 0:
                root = 0  # Default to node 0
            else:
                root = int(root_candidates[0])

        # Create tree
        tree = cls(n_nodes=n_nodes, root=root)

        # Build tree structure using BFS from root
        from collections import deque
        visited = set([root])
        queue = deque([root])

        while queue:
            parent = queue.popleft()
            # Find children (outgoing edges from parent)
            children = np.where(adj[parent, :] > 0)[0]

            for child in children:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

                    # Get edge weight
                    if weighted:
                        length = float(adj[parent, child])
                    else:
                        length = 1.0

                    tree.add_edge(parent, int(child), length=length)

        return tree

    @classmethod
    def from_parent_array(cls, parent_array: np.ndarray | list[int],
                         edge_lengths: Optional[np.ndarray] = None,
                         root: Optional[int] = None) -> 'Tree':
        """
        Create tree from parent array (MATLAB-style).
        
        Parameters
        ----------
        parent_array : np.ndarray
            Array where parent_array[i] is the parent of node i.
            Root node should have parent -1 or point to itself.
        edge_lengths : np.ndarray, optional
            Array of edge lengths. If None, all edges have length 1.
        root : int, optional
            Root node index. If None, will be inferred.
        
        Returns
        -------
        Tree instance
        """
        if isinstance(parent_array, list):
            try:
                parent_array = np.array(parent_array)
            except TypeError:
                raise TypeError(f"Cannot convert parent_array type '{type(parent_array).__name__}' to a NumPy array.")

        n_nodes = len(parent_array)
        
        # Find root if not specified
        if root is None:
            root_candidates = np.where((parent_array == -1) | 
                                      (parent_array == np.arange(n_nodes)))[0]
            if len(root_candidates) == 0:
                raise ValueError("No root found in parent array")
            root = int(root_candidates[0])
        
        tree = cls(n_nodes=n_nodes, root=root)
        
        # Build tree from parent array
        for child in range(n_nodes):
            if child != root:
                parent = int(parent_array[child])
                if parent == child:  # Sometimes root points to itself
                    continue
                length = 1.0 if edge_lengths is None else float(edge_lengths[child])
                tree.add_edge(parent, child, length=length)
        
        return tree
    
    def to_parent_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export tree as parent array (MATLAB-style).
        
        Returns
        -------
        parent_array : np.ndarray
            Parent array where parent_array[i] is the parent of node i
        edge_lengths : np.ndarray
            Array of edge lengths
        """
        edge_lengths_array = np.ones(self.n_nodes)
        
        for i in range(self.n_nodes):
            if not self.is_root(i):
                parent = self.parent[i]
                edge = (parent, i)
                edge_lengths_array[i] = self.edge_lengths.get(edge, 1.0)
        
        return self.parent.copy(), edge_lengths_array
    
    def get_edge_length(self, parent: int, child: int) -> float:
        """Get length of edge between parent and child."""
        return self.edge_lengths.get((parent, child), 1.0)
    
    def total_length(self) -> float:
        """Get total length of all edges in tree."""
        return sum(self.edge_lengths.values())
    
    @classmethod
    def from_networkx(cls, G):
        """
        Create tree from NetworkX graph.
        
        Automatically detects root node and rebuilds tree structure.
        """
        import networkx as nx
        from collections import deque
        
        n_nodes = G.number_of_nodes()
        
        # Find root (node with no incoming edges)
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        if len(roots) != 1:
            # If multiple roots or no root, use node 0
            root = 0
        else:
            root = roots[0]
        
        instance = cls(n_nodes=n_nodes, root=root)
        
        # Rebuild tree structure using BFS from root
        visited = set()
        queue = deque([root])
        visited.add(root)
        
        while queue:
            parent = queue.popleft()
            for child in G.successors(parent):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    # Get edge attributes
                    edge_data = G.get_edge_data(parent, child)
                    length = edge_data.get('length', 1.0) if edge_data else 1.0
                    # Add edge
                    instance.add_edge(parent, child, length=length)
        
        return instance
    
    @classmethod
    def from_igraph(cls, g):
        """
        Create tree from igraph graph.
        
        Automatically detects root and rebuilds tree structure.
        """
        from collections import deque
        
        n_nodes = g.vcount()
        
        # Find root (node with no incoming edges)
        in_degrees = g.indegree()
        roots = [i for i, deg in enumerate(in_degrees) if deg == 0]
        if len(roots) != 1:
            root = 0
        else:
            root = roots[0]
        
        instance = cls(n_nodes=n_nodes, root=root)
        
        # Rebuild tree structure using BFS from root
        visited = set()
        queue = deque([root])
        visited.add(root)
        
        while queue:
            parent = queue.popleft()
            # Get outgoing edges from parent
            out_edges = g.es.select(_source=parent)
            for edge in out_edges:
                child = edge.target
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    # Get edge length
                    length = edge['length'] if 'length' in edge.attributes() else 1.0
                    instance.add_edge(parent, child, length=length)
        
        return instance
    
    def __repr__(self):
        leaves = len(self.get_leaves())
        return (f"{self.__class__.__name__}(n_nodes={self.n_nodes}, "
                f"n_leaves={leaves}, root={self.root})")
