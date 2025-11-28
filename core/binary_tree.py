"""
Binary tree data structure with multi-library compatibility.

Extends Tree with binary-specific constraints and operations.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from .tree import Tree


class BinaryTree(Tree):
    """
    Binary tree where each node has at most 2 children.
    
    Maintains explicit left/right child tracking for binary-specific operations.
    Also stores whether the tree is 'planted' (root has degree 1).
    """
    
    def __init__(self, n_nodes: int = 0, root: Optional[int] = None, planted: bool = True):
        super().__init__(n_nodes, root)
        
        # Binary tree specific structures
        self.left_child: np.ndarray = np.full(n_nodes, -1, dtype=np.int32)
        self.right_child: np.ndarray = np.full(n_nodes, -1, dtype=np.int32)
        self.planted = planted  # Whether root has degree 1
    
    def add_edge(self, parent: int, child: int, side: Optional[str] = None, 
                 length: float = 1.0, **attrs):
        """
        Add an edge from parent to child.
        
        Parameters
        ----------
        parent : int
            Parent node index
        child : int
            Child node index
        side : str, optional
            'left' or 'right'. If None, will be determined automatically.
        length : float
            Edge length
        **attrs
            Additional edge attributes
        """
        # Validate binary constraint
        if len(self.children[parent]) >= 2:
            raise ValueError(f"Node {parent} already has 2 children. Binary tree constraint violated.")
        
        # Determine side if not specified
        if side is None:
            if self.left_child[parent] == -1:
                side = 'left'
            elif self.right_child[parent] == -1:
                side = 'right'
            else:
                raise ValueError(f"Node {parent} already has 2 children")
        
        # Update left/right tracking
        if side == 'left':
            if self.left_child[parent] != -1:
                raise ValueError(f"Node {parent} already has a left child")
            self.left_child[parent] = child
        elif side == 'right':
            if self.right_child[parent] != -1:
                raise ValueError(f"Node {parent} already has a right child")
            self.right_child[parent] = child
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")
        
        # Call parent add_edge
        super().add_edge(parent, child, length=length, **attrs)
        
        # Store side as edge attribute
        self._graph.edge_attrs[(parent, child)]['side'] = side

    
    def get_left_child(self, node: int) -> int:
        """Get left child of node. Returns -1 if no left child."""
        return int(self.left_child[node])
    
    def get_right_child(self, node: int) -> int:
        """Get right child of node. Returns -1 if no right child."""
        return int(self.right_child[node])
    
    def is_left_child(self, node: int) -> bool:
        """Check if node is a left child of its parent."""
        if self.is_root(node):
            return False
        parent = self.parent[node]
        return self.left_child[parent] == node
    
    def is_right_child(self, node: int) -> bool:
        """Check if node is a right child of its parent."""
        if self.is_root(node):
            return False
        parent = self.parent[node]
        return self.right_child[parent] == node
    
    def get_sibling(self, node: int) -> int:
        """Get sibling of node. Returns -1 if no sibling or node is root."""
        if self.is_root(node):
            return -1
        
        parent = self.parent[node]
        if self.is_left_child(node):
            return self.right_child[parent]
        else:
            return self.left_child[parent]
    
    def depth_first_search_lr(self, start: Optional[int] = None) -> List[int]:
        """
        Depth-first search with explicit left-right ordering.
        
        Returns nodes in order: leftmost leaf → root → rightmost leaf
        (standard DFS for binary trees).
        """
        if start is None:
            start = self.root
        
        order = []
        
        def dfs(node):
            if node == -1:
                return
            dfs(self.left_child[node])
            order.append(node)
            dfs(self.right_child[node])
        
        dfs(start)
        return order
    
    def get_leaves_lr(self) -> List[int]:
        """Get leaves in left-to-right order."""
        return [node for node in self.depth_first_search_lr() if self.is_leaf(node)]
    
    def validate(self) -> bool:
        """
        Validate binary tree structure.
        
        Checks Tree constraints plus:
        - Each node has at most 2 children
        - Left/right child arrays are consistent with children lists
        """
        # Check parent tree validation
        if not super().validate():
            return False
        
        # Check binary constraint
        for node in range(self.n_nodes):
            if len(self.children[node]) > 2:
                return False
            
            # Check consistency of left/right with children list
            left = self.left_child[node]
            right = self.right_child[node]
            children_set = set(self.children[node])
            
            expected_children = set()
            if left != -1:
                expected_children.add(left)
            if right != -1:
                expected_children.add(right)
            
            if children_set != expected_children:
                return False
        
        return True
    
    @classmethod
    def from_parent_array(cls, parent_array: np.ndarray,
                         edge_lengths: Optional[np.ndarray] = None,
                         left_right_order: Optional[List[Tuple[int, str]]] = None,
                         root: Optional[int] = None,
                         planted: bool = True) -> 'BinaryTree':
        """
        Create binary tree from parent array.
        
        Parameters
        ----------
        parent_array : np.ndarray
            Parent array where parent_array[i] is parent of node i
        edge_lengths : np.ndarray, optional
            Edge lengths
        left_right_order : List[Tuple[int, str]], optional
            List of (node, side) tuples specifying whether each child is 'left' or 'right'.
            If None, will assign left/right based on index order.
        root : int, optional
            Root node index
        planted : bool
            Whether the tree is planted (root has degree 1)
        
        Returns
        -------
        BinaryTree instance
        """
        n_nodes = len(parent_array)
        
        # Find root if not specified
        if root is None:
            root_candidates = np.where((parent_array == -1) | 
                                      (parent_array == np.arange(n_nodes)))[0]
            if len(root_candidates) == 0:
                raise ValueError("No root found in parent array")
            root = int(root_candidates[0])
        
        tree = cls(n_nodes=n_nodes, root=root, planted=planted)
        
        # Build mapping of which children are left/right
        if left_right_order is None:
            # Default: first child is left, second is right (by index)
            parent_children = {i: [] for i in range(n_nodes)}
            for child in range(n_nodes):
                if child != root and parent_array[child] != child:
                    parent = int(parent_array[child])
                    parent_children[parent].append(child)
            
            # Sort children by index for consistent left/right assignment
            for parent in parent_children:
                parent_children[parent].sort()
        
        # Add edges
        for child in range(n_nodes):
            if child != root and parent_array[child] != child:
                parent = int(parent_array[child])
                length = 1.0 if edge_lengths is None else float(edge_lengths[child])
                
                # Determine side
                if left_right_order is not None:
                    # Use provided left/right info
                    side = next((s for n, s in left_right_order if n == child), None)
                    if side is None:
                        # Guess based on existing children
                        side = 'left' if tree.left_child[parent] == -1 else 'right'
                else:
                    # Use index-based ordering
                    children_of_parent = parent_children[parent]
                    idx = children_of_parent.index(child)
                    side = 'left' if idx == 0 else 'right'
                
                tree.add_edge(parent, child, side=side, length=length)
        
        return tree
    
    def to_parent_array_with_sides(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]]]:
        """
        Export as parent array with left/right information.
        
        Returns
        -------
        parent_array : np.ndarray
        edge_lengths : np.ndarray
        left_right_info : List[Tuple[int, str]]
            List of (node, side) tuples
        """
        parent_arr, lengths = super().to_parent_array()
        
        left_right_info = []
        for node in range(self.n_nodes):
            if not self.is_root(node):
                side = 'left' if self.is_left_child(node) else 'right'
                left_right_info.append((node, side))
        
        return parent_arr, lengths, left_right_info
    
    def __repr__(self):
        leaves = len(self.get_leaves())
        max_depth = np.max(self.get_depth()) if self.n_nodes > 0 else 0
        return (f"BinaryTree(n_nodes={self.n_nodes}, n_leaves={leaves}, "
                f"max_depth={max_depth}, planted={self.planted})")
