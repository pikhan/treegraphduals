"""
Test suite for Binary Tree class.

Run with:
    pytest tests/test_binary_tree.py
    pytest tests/test_binary_tree.py -v
    pytest tests/test_binary_tree.py --cov=core.binary_tree
"""

import pytest
import numpy as np
import importlib.util

from core.binary_tree import BinaryTree

class TestBasicBinaryTree:
    """Test some basic binary tree methods"""

    def test_demo_binary_tree(self):
        """Demonstrate binary tree construction and operations."""

        # Create a binary tree
        btree = BinaryTree(n_nodes=8, root=0, planted=False)

        # Build binary tree structure:
        #         0
        #        / \
        #       1   2
        #      / \
        #     3   4
        #    /   / \
        #   5   6   7

        btree.add_edge(0, 1, side='left', length=0.5)
        btree.add_edge(1, 3, side='left', length=2.5)
        btree.add_edge(1, 4, side='right', length=1.5)
        btree.add_edge(0, 2, side='right', length=4.0)
        btree.add_edge(3, 5, side='left', length=1.0)
        btree.add_edge(4, 6, side='left', length=2.0)
        btree.add_edge(4, 7, length=3.0)

        #print(f"\nBinary Tree: {btree}")
        assert btree.validate()
        assert [int(x) for x in btree.get_leaves_lr()] == [5, 6, 7, 2]
        assert [value for key, value in sorted(btree.horton_strahler_order().items())] == [2, 2, 1, 1, 2, 1, 1, 1]
        assert btree.get_left_child(1) == 3
        assert btree.get_right_child(1) == 4
        assert btree.is_left_child(2) == False
        assert btree.is_left_child(5) == True
        assert btree.is_right_child(4) == True
        assert btree.is_right_child(6) == False
        assert btree.get_sibling(2) == 1
        assert btree.get_sibling(0) == -1

    def test_parent_array_conversion(self):
        """Demonstrate conversion from MATLAB-style parent arrays."""

        # MATLAB-style parent array
        # Node 0 is root (parent = -1)
        parent_array = np.array([-1, 0, 0, 1, 1, 2, 2])
        edge_lengths = np.array([0, 1.0, 1.5, 2.0, 1.0, 1.0, 0.5])

        # Create tree from parent array
        tree = BinaryTree.from_parent_array(parent_array, edge_lengths)
        assert tree.validate() == True

        # Convert back to parent array
        parent_back, lengths_back = tree.to_parent_array()

        # Verify round-trip
        assert np.allclose(parent_array, parent_back), "Parent array mismatch!"

        # Edge lengths for non-root nodes should match (root node has no incoming edge)
        non_root_mask = parent_array != -1
        assert np.allclose(edge_lengths[non_root_mask], lengths_back[non_root_mask]), "Edge lengths mismatch!"

    def test_networkx_conversion(self):
        """Demonstrate NetworkX conversion."""

        # Create tree
        tree = BinaryTree(n_nodes=5, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(1, 4, length=1.0)

        # Convert to NetworkX
        nx_graph = tree.to_networkx(directed=True)
        # print(f"Nodes: {list(nx_graph.nodes())}")
        # print(f"Edges: {list(nx_graph.edges(data=True))}")

        # Convert back from NetworkX
        tree_back = BinaryTree.from_networkx(nx_graph)
        assert tree_back.validate() == True

        # Verify structure preserved
        assert tree.n_nodes == tree_back.n_nodes, "Node count mismatch!"
        assert tree.n_edges == tree_back.n_edges, "Edge count mismatch!"


    def test_igraph_conversion(self):
        """Demonstrate igraph conversion."""

        # Create tree
        tree = BinaryTree(n_nodes=5, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(1, 4, length=1.0)


        # Convert to igraph
        ig_graph = tree.to_igraph(directed=True)
        # print(f"\nigraph graph: {ig_graph}")
        # print(f"Vertices: {ig_graph.vcount()}")
        # print(f"Edges: {ig_graph.ecount()}")
        # print(f"Edge lengths: {ig_graph.es['length']}")

        # Convert back from igraph
        tree_back = BinaryTree.from_igraph(ig_graph)
        # print(f"\nConverted back: {tree_back}")
        assert tree_back.validate() == True

        # Verify structure preserved
        assert tree.n_nodes == tree_back.n_nodes, "Node count mismatch!"
        assert tree.n_edges == tree_back.n_edges, "Edge count mismatch!"

    def test_adjacency_matrix(self):
        """Demonstrate adjacency matrix conversion."""

        # Create tree
        tree = BinaryTree(n_nodes=5, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(1, 4, length=1.0)

        # Convert to adjacency matrix (weighted)
        adj_weighted = tree.to_adjacency_matrix(weighted=True, weight_attr='length')

        # Create tree from adjacency matrix
        tree_back = BinaryTree.from_adjacency_matrix(adj_weighted, weighted=True, root=0)

        # Test structural equality instead of object equality
        assert tree_back.n_nodes == tree.n_nodes
        assert tree_back.n_edges == tree.n_edges
        assert tree_back.root == tree.root

        # Check parent structure matches
        for i in range(tree.n_nodes):
            assert tree_back.parent[i] == tree.parent[i]

        # Check edge lengths match
        for i in range(tree.n_nodes):
            if not tree.is_root(i):
                parent = int(tree.parent[i])
                assert tree_back.get_edge_length(parent, i) == tree.get_edge_length(parent, i)

        # Check leaves match
        assert set(tree_back.get_leaves()) == set(tree.get_leaves())

if __name__ == '__main__':
    pytest.main([__file__, '-v'])