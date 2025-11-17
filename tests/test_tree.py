"""
Test suite for Tree class.

Run with:
    pytest tests/test_tree.py
    pytest tests/test_tree.py -v
    pytest tests/test_tree.py --cov=core.tree
"""

import pytest
import numpy as np
import sys

from core.tree import Tree


class TestTreeConstruction:
    """Test tree construction methods."""
    
    def test_empty_tree(self):
        """Test creating an empty tree."""
        tree = Tree(n_nodes=0)
        assert tree.n_nodes == 0
        assert tree.n_edges == 0
        assert tree.validate()
    
    def test_basic_construction(self):
        """Test basic tree construction with edges."""
        tree = Tree(n_nodes=5, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(1, 4, length=1.0)
        
        assert tree.n_nodes == 5
        assert tree.n_edges == 4
        assert tree.root == 0
        assert tree.validate()
    
    def test_from_parent_array(self):
        """Test construction from parent array."""
        parent = np.array([-1, 0, 0, 1, 1])
        lengths = np.array([0, 1.0, 1.5, 2.0, 1.0])
        
        tree = Tree.from_parent_array(parent, lengths)
        
        assert tree.n_nodes == 5
        assert tree.n_edges == 4
        assert tree.validate()
    
    def test_parent_array_round_trip(self):
        """Test that parent array conversion is reversible."""
        original_parent = np.array([-1, 0, 0, 1, 1])
        original_lengths = np.array([0, 1.0, 1.5, 2.0, 1.0])
        
        tree = Tree.from_parent_array(original_parent, original_lengths)
        parent_back, lengths_back = tree.to_parent_array()
        
        np.testing.assert_array_equal(original_parent, parent_back)
        # Check non-root nodes
        mask = original_parent != -1
        np.testing.assert_array_almost_equal(
            original_lengths[mask], 
            lengths_back[mask]
        )


class TestTreeOperations:
    """Test tree operations and queries."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing.
        
        Structure:
              0
             / \
            1   2
           /   / \
          3   4   5
         /
        6
        """
        tree = Tree(n_nodes=7, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(2, 4, length=1.0)
        tree.add_edge(2, 5, length=1.0)
        tree.add_edge(3, 6, length=0.5)
        return tree
    
    def test_get_parent(self, sample_tree):
        """Test parent retrieval."""
        assert sample_tree.get_parent(0) == -1  # root has no parent
        assert sample_tree.get_parent(1) == 0
        assert sample_tree.get_parent(3) == 1
        assert sample_tree.get_parent(6) == 3
    
    def test_get_children(self, sample_tree):
        """Test children retrieval."""
        assert set(sample_tree.get_children(0)) == {1, 2}
        assert set(sample_tree.get_children(1)) == {3}
        assert set(sample_tree.get_children(2)) == {4, 5}
        assert sample_tree.get_children(6) == []
    
    def test_is_leaf(self, sample_tree):
        """Test leaf identification."""
        assert not sample_tree.is_leaf(0)
        assert not sample_tree.is_leaf(1)
        assert sample_tree.is_leaf(4)
        assert sample_tree.is_leaf(5)
        assert sample_tree.is_leaf(6)
    
    def test_get_leaves(self, sample_tree):
        """Test getting all leaves."""
        leaves = sample_tree.get_leaves()
        assert set(leaves) == {4, 5, 6}
    
    def test_dfs(self, sample_tree):
        """Test depth-first search."""
        dfs_order = sample_tree.depth_first_search()
        assert len(dfs_order) == 7
        assert dfs_order[0] == 0  # Start at root
        # Check parent comes before children
        parent_positions = {node: i for i, node in enumerate(dfs_order)}
        for node in range(1, 7):
            parent = sample_tree.get_parent(node)
            assert parent_positions[parent] < parent_positions[node]
    
    def test_bfs(self, sample_tree):
        """Test breadth-first search."""
        bfs_order = sample_tree.breadth_first_search()
        assert len(bfs_order) == 7
        assert bfs_order[0] == 0  # Start at root
        # Check levels are correct
        assert bfs_order[1] in {1, 2}
        assert bfs_order[2] in {1, 2}
    
    def test_get_depth(self, sample_tree):
        """Test depth computation."""
        assert sample_tree.get_depth(0) == 0
        assert sample_tree.get_depth(1) == 1
        assert sample_tree.get_depth(3) == 2
        assert sample_tree.get_depth(6) == 3
        
        # Test getting all depths
        depths = sample_tree.get_depth()
        expected = np.array([0, 1, 1, 2, 2, 2, 3])
        np.testing.assert_array_equal(depths, expected)
    
    def test_get_subtree_size(self, sample_tree):
        """Test subtree size computation."""
        assert sample_tree.get_subtree_size(0) == 7
        assert sample_tree.get_subtree_size(1) == 3  # nodes 1, 3, 6
        assert sample_tree.get_subtree_size(2) == 3  # nodes 2, 4, 5
        assert sample_tree.get_subtree_size(6) == 1  # just itself
    
    def test_get_path_to_root(self, sample_tree):
        """Test path to root."""
        path = sample_tree.get_path_to_root(6)
        assert path == [6, 3, 1, 0]
        
        path = sample_tree.get_path_to_root(5)
        assert path == [5, 2, 0]
    
    def test_get_path_between(self, sample_tree):
        """Test path between two nodes."""
        path = sample_tree.get_path_between(6, 5)
        # Path should go: 6 -> 3 -> 1 -> 0 -> 2 -> 5
        assert path[0] == 6
        assert path[-1] == 5
        assert 0 in path  # Must go through root
    
    def test_get_distance_unweighted(self, sample_tree):
        """Test unweighted distance (edge count)."""
        dist = sample_tree.get_distance(6, 5, weighted=False)
        assert dist == 5  # 5 edges between them
    
    def test_get_distance_weighted(self, sample_tree):
        """Test weighted distance."""
        dist = sample_tree.get_distance(6, 5, weighted=True)
        # Path: 6 -> 3 -> 1 -> 0 -> 2 -> 5
        # Lengths: 0.5 + 2.0 + 1.0 + 1.5 + 1.0 = 6.0
        assert dist == 6.0


class TestTreeConversions:
    """Test tree conversions to other formats."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a small tree for conversion tests."""
        tree = Tree(n_nodes=5, root=0)
        tree.add_edge(0, 1, length=1.0)
        tree.add_edge(0, 2, length=1.5)
        tree.add_edge(1, 3, length=2.0)
        tree.add_edge(1, 4, length=1.0)
        return tree
    
    def test_to_adjacency_matrix_binary(self, sample_tree):
        """Test conversion to binary adjacency matrix."""
        adj = sample_tree.to_adjacency_matrix(weighted=False)
        
        assert adj.shape == (5, 5)
        assert adj[0, 1] == 1
        assert adj[0, 2] == 1
        assert adj[1, 3] == 1
        assert adj[1, 4] == 1
        assert adj[3, 0] == 0  # No edge from 3 to 0
    
    def test_to_adjacency_matrix_weighted(self, sample_tree):
        """Test conversion to weighted adjacency matrix."""
        adj = sample_tree.to_adjacency_matrix(weighted=True)
        
        assert adj.shape == (5, 5)
        assert adj[0, 1] == 1.0
        assert adj[0, 2] == 1.5
        assert adj[1, 3] == 2.0
        assert adj[1, 4] == 1.0
    
    def test_from_adjacency_matrix(self, sample_tree):
        """Test round-trip through adjacency matrix."""
        adj = sample_tree.to_adjacency_matrix(weighted=True)
        tree_back = Tree.from_adjacency_matrix(adj, weighted=True)
        
        assert tree_back.n_nodes == sample_tree.n_nodes
        assert tree_back.n_edges == sample_tree.n_edges
    
    @pytest.mark.skipif(
        'networkx' not in sys.modules,
        reason="NetworkX not installed"
    )
    def test_to_networkx(self, sample_tree):
        """Test conversion to NetworkX."""
        import networkx as nx
        
        nx_graph = sample_tree.to_networkx(directed=True)
        
        assert isinstance(nx_graph, nx.DiGraph)
        assert nx_graph.number_of_nodes() == 5
        assert nx_graph.number_of_edges() == 4
    
    @pytest.mark.skipif(
        'networkx' not in sys.modules,
        reason="NetworkX not installed"
    )
    def test_from_networkx(self, sample_tree):
        """Test round-trip through NetworkX."""
        import networkx as nx
        
        nx_graph = sample_tree.to_networkx(directed=True)
        tree_back = Tree.from_networkx(nx_graph)
        
        assert tree_back.n_nodes == sample_tree.n_nodes
        assert tree_back.n_edges == sample_tree.n_edges
        assert tree_back.validate()


class TestTreeValidation:
    """Test tree validation."""
    
    def test_valid_tree(self):
        """Test that valid trees pass validation."""
        tree = Tree(n_nodes=3, root=0)
        tree.add_edge(0, 1)
        tree.add_edge(0, 2)
        assert tree.validate()
    
    def test_disconnected_tree_invalid(self):
        """Test that disconnected trees fail validation."""
        tree = Tree(n_nodes=4, root=0)
        tree.add_edge(0, 1)
        # Nodes 2 and 3 are disconnected
        tree.add_edge(2, 3)
        assert not tree.validate()
    
    def test_multiple_roots_invalid(self):
        """Test that trees with multiple roots fail validation."""
        tree = Tree(n_nodes=3, root=0)
        # Manually create invalid state with no edges
        # (multiple isolated nodes, each is its own root)
        assert not tree.validate()


class TestTreeEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_node_tree(self):
        """Test tree with single node."""
        tree = Tree(n_nodes=1, root=0)
        assert tree.n_nodes == 1
        assert tree.n_edges == 0
        assert tree.validate()
        assert tree.get_leaves() == [0]
    
    def test_node_attributes(self):
        """Test adding node attributes."""
        tree = Tree(n_nodes=3, root=0)
        tree.add_edge(0, 1)
        tree.add_edge(0, 2)
        
        tree.add_node_attr(1, 'label', 'node_A')
        tree.add_node_attr(1, 'weight', 5.0)
        
        # Verify attributes are stored
        assert tree._graph.node_attrs[1]['label'] == 'node_A'
        assert tree._graph.node_attrs[1]['weight'] == 5.0
    
    def test_edge_attributes(self):
        """Test edge attributes beyond length."""
        tree = Tree(n_nodes=2, root=0)
        tree.add_edge(0, 1, length=1.0, color='red', label='edge_A')
        
        # Verify attributes are stored
        attrs = tree._graph.edge_attrs[(0, 1)]
        assert attrs['length'] == 1.0
        assert attrs['color'] == 'red'
        assert attrs['label'] == 'edge_A'


# Parametrized tests
class TestTreeParametrized:
    """Parametrized tests for different tree configurations."""
    
    @pytest.mark.parametrize("n_nodes,expected_max_depth", [
        (1, 0),
        (2, 1),
        (3, 1),
        (7, 2),
    ])
    def test_balanced_tree_depth(self, n_nodes, expected_max_depth):
        """Test depth of balanced trees."""
        tree = Tree(n_nodes=n_nodes, root=0)
        for i in range(1, n_nodes):
            parent = (i - 1) // 2
            tree.add_edge(parent, i)
        
        depths = tree.get_depth()
        assert np.max(depths) == expected_max_depth
    
    @pytest.mark.parametrize("n_nodes", [10, 50, 100, 500])
    @pytest.mark.slow
    def test_large_tree_operations(self, n_nodes):
        """Test operations scale with tree size."""
        # Create chain tree
        tree = Tree(n_nodes=n_nodes, root=0)
        for i in range(1, n_nodes):
            tree.add_edge(i - 1, i)
        
        assert tree.validate()
        assert len(tree.depth_first_search()) == n_nodes
        assert len(tree.get_leaves()) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
