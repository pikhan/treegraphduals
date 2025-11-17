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
        
        # Caches for binary tree metrics
        self._horton_order_cache: Optional[np.ndarray] = None
    
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
        
        # Invalidate binary tree caches
        self._horton_order_cache = None
    
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
    
    def horton_prune(self) -> 'BinaryTree':
        """
        Perform one Horton pruning operation.
        
        Removes all leaves and performs series reduction (removes degree-2 nodes).
        
        Returns
        -------
        BinaryTree
            The pruned tree
        """
        # Identify leaves to remove
        leaves = self.get_leaves()
        
        if len(leaves) == self.n_nodes:
            # Tree is all leaves, return empty tree
            return BinaryTree(n_nodes=0)
        
        # Build new tree without leaves
        old_to_new = {}
        new_idx = 0
        
        # First pass: assign new indices to non-leaf nodes
        for node in range(self.n_nodes):
            if not self.is_leaf(node):
                old_to_new[node] = new_idx
                new_idx += 1
        
        if new_idx == 0:
            return BinaryTree(n_nodes=0)
        
        new_tree = BinaryTree(n_nodes=new_idx, root=old_to_new.get(self.root, 0))
        
        # Second pass: rebuild tree structure
        for old_node in range(self.n_nodes):
            if old_node not in old_to_new:
                continue
            
            new_node = old_to_new[old_node]
            
            # Process children
            left = self.left_child[old_node]
            right = self.right_child[old_node]
            
            # Add non-leaf children or perform series reduction
            if left != -1 and left not in leaves:
                new_left = old_to_new[left]
                length = self.edge_lengths.get((old_node, left), 1.0)
                
                # Series reduction: if left child has only one child, merge edges
                while (new_left in old_to_new and 
                       len([c for c in new_tree.children[new_left] if c != -1]) == 1 and
                       not new_tree.is_root(new_left)):
                    # Find the single child
                    single_child = (new_tree.left_child[new_left] if new_tree.left_child[new_left] != -1 
                                  else new_tree.right_child[new_left])
                    if single_child == -1:
                        break
                    length += new_tree.edge_lengths.get((new_left, single_child), 1.0)
                    new_left = single_child
                
                if new_left != -1:
                    new_tree.add_edge(new_node, new_left, side='left', length=length)
            
            if right != -1 and right not in leaves:
                new_right = old_to_new[right]
                length = self.edge_lengths.get((old_node, right), 1.0)
                
                # Series reduction for right child
                while (new_right in old_to_new and 
                       len([c for c in new_tree.children[new_right] if c != -1]) == 1 and
                       not new_tree.is_root(new_right)):
                    single_child = (new_tree.left_child[new_right] if new_tree.left_child[new_right] != -1 
                                  else new_tree.right_child[new_right])
                    if single_child == -1:
                        break
                    length += new_tree.edge_lengths.get((new_right, single_child), 1.0)
                    new_right = single_child
                
                if new_right != -1:
                    new_tree.add_edge(new_node, new_right, side='right', length=length)
        
        return new_tree
    
    def horton_strahler_order(self, node: Optional[int] = None) -> int:
        """
        Compute Horton-Strahler order of a node or the entire tree.
        
        Parameters
        ----------
        node : int, optional
            If provided, return order of this node.
            If None, return order of the tree.
        
        Returns
        -------
        int
            Horton-Strahler order
        """
        if self._horton_order_cache is None:
            orders = np.zeros(self.n_nodes, dtype=np.int32)
            
            # Compute orders bottom-up
            for n in reversed(self.depth_first_search()):
                if self.is_leaf(n):
                    orders[n] = 1
                else:
                    left = self.left_child[n]
                    right = self.right_child[n]
                    
                    left_order = orders[left] if left != -1 else 0
                    right_order = orders[right] if right != -1 else 0
                    
                    if left_order == right_order and left_order > 0:
                        orders[n] = left_order + 1
                    else:
                        orders[n] = max(left_order, right_order)
            
            self._horton_order_cache = orders
        
        if node is not None:
            return int(self._horton_order_cache[node])
        else:
            return int(self._horton_order_cache[self.root])
    
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
