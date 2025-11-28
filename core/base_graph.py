"""
Generic representation and base graph classes with multi-library compatibility.

Supports NetworkX, igraph, numpy arrays, scipy sparse matrices, and other representations.
BaseGraph(ABC)
├── DAG
│   └── Polytree
├── Tree
│   └── BinaryTree
│       ├── GaltonWatsonTree
│   └── Forest
├── GeneralGraph
│   ├── Multigraph
│   └── ErdosRenyi
└── RealTree (separate - different math)
"""

import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Any
from abc import ABC, abstractmethod

class GraphRepresentation:
    """
    Lightweight internal representation of a graph.
    
    Stores graph data in an efficient format and provides conversions
    to networkx, igraph, and scipy graph libraries and various graph representations.
    """
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.edges: List[Tuple[int, int]] = []
        self.edge_attrs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.node_attrs: Dict[int, Dict[str, Any]] = {i: {} for i in range(n_nodes)}
        
        # Cache for converted representations
        self._networkx_cache = None
        self._igraph_cache = None
        self._adjacency_cache = None
        self._dirty = False  # Track if graph has been modified
    
    def add_edge(self, u: int, v: int, **attrs):
        """Add an edge with optional attributes."""
        edge = (u, v)
        if edge not in self.edges:
            self.edges.append(edge)
        self.edge_attrs[edge] = attrs
        self._invalidate_caches()
    
    def add_node_attr(self, node: int, key: str, value: Any):
        """Add attribute to a node."""
        self.node_attrs[node][key] = value
        self._invalidate_caches()
    
    def _invalidate_caches(self):
        """Invalidate cached representations when graph is modified."""
        self._networkx_cache = None
        self._igraph_cache = None
        self._adjacency_cache = None
        self._dirty = True
    
    def to_networkx(self, directed: bool = True):
        """Convert to NetworkX graph."""
        if self._networkx_cache is not None and not self._dirty:
            return self._networkx_cache
        
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for this conversion. Install with: pip install networkx or per your package manager's syntax")
        
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Add node attributes
        for node, attrs in self.node_attrs.items():
            for key, value in attrs.items():
                G.nodes[node][key] = value
        
        # Add edges with attributes
        for (u, v) in self.edges:
            attrs = self.edge_attrs.get((u, v), {})
            G.add_edge(u, v, **attrs)
        
        self._networkx_cache = G
        self._dirty = False
        return G
    
    def to_igraph(self, directed: bool = True):
        """Convert to igraph graph."""
        if self._igraph_cache is not None and not self._dirty:
            return self._igraph_cache
        
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("igraph is required for this conversion. Install with: pip install igraph or per your package manager's syntax")

        g = ig.Graph(n=self.n_nodes, directed=directed)
        
        # Add edges
        if self.edges:
            g.add_edges(self.edges)
        
        # Add edge attributes
        for edge_idx, (u, v) in enumerate(self.edges):
            attrs = self.edge_attrs.get((u, v), {})
            for key, value in attrs.items():
                if key not in g.es.attributes():
                    g.es[key] = [None] * g.ecount()
                g.es[edge_idx][key] = value
        
        # Add node attributes
        for node, attrs in self.node_attrs.items():
            for key, value in attrs.items():
                if key not in g.vs.attributes():
                    g.vs[key] = [None] * g.vcount()
                g.vs[node][key] = value
        
        self._igraph_cache = g
        self._dirty = False
        return g
    
    def to_adjacency_matrix(self, weighted: bool = False, 
                           weight_attr: str = 'length') -> np.ndarray:
        """
        Convert to adjacency matrix (numpy array).
        
        Parameters
        ----------
        weighted : bool
            If True, use edge weights. If False, binary adjacency.
        weight_attr : str
            Edge attribute to use for weights.
        """
        if not weighted and self._adjacency_cache is not None:
            return self._adjacency_cache
        
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        for (u, v) in self.edges:
            if weighted and weight_attr in self.edge_attrs.get((u, v), {}):
                adj[u, v] = self.edge_attrs[(u, v)][weight_attr]
            else:
                adj[u, v] = 1
        
        if not weighted:
            self._adjacency_cache = adj
        
        return adj
    
    def to_sparse_matrix(self, weighted: bool = False, weight_attr: str = 'length'):
        """Convert to scipy sparse matrix (CSR format)."""
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError("scipy is required for sparse matrices. Install with: pip install scipy or per your package manager's syntax")
        
        adj = self.to_adjacency_matrix(weighted=weighted, weight_attr=weight_attr)
        return csr_matrix(adj)
    
    @classmethod
    def from_networkx(cls, G):
        """Create from NetworkX graph."""
        n_nodes = G.number_of_nodes()
        graph_rep = cls(n_nodes)
        
        # Add edges with attributes
        for u, v, attrs in G.edges(data=True):
            graph_rep.add_edge(u, v, **attrs)
        
        # Add node attributes
        for node, attrs in G.nodes(data=True):
            for key, value in attrs.items():
                graph_rep.add_node_attr(node, key, value)
        
        return graph_rep
    
    @classmethod
    def from_igraph(cls, g):
        """Create from igraph graph."""
        n_nodes = g.vcount()
        graph_rep = cls(n_nodes)
        
        # Add edges with attributes
        for edge in g.es:
            u, v = edge.tuple
            attrs = {key: edge[key] for key in edge.attributes()}
            graph_rep.add_edge(u, v, **attrs)
        
        # Add node attributes
        for node_idx in range(n_nodes):
            vertex = g.vs[node_idx]
            for key in vertex.attributes():
                graph_rep.add_node_attr(node_idx, key, vertex[key])
        
        return graph_rep
    
    @classmethod
    def from_adjacency_matrix(cls, adj: np.ndarray, weighted: bool = False):
        """Create from adjacency matrix."""
        n_nodes = adj.shape[0]
        graph_rep = cls(n_nodes)
        
        rows, cols = np.nonzero(adj)
        for u, v in zip(rows, cols):
            if weighted:
                graph_rep.add_edge(int(u), int(v), weight=float(adj[u, v]))
            else:
                graph_rep.add_edge(int(u), int(v))
        
        return graph_rep


class BaseGraph(ABC):
    """
    Abstract base class for all graph structures.
    
    This provides a unified interface while maintaining compatibility
    with multiple graph libraries.
    """
    
    def __init__(self, n_nodes: int = 0):
        self._graph = GraphRepresentation(n_nodes)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.n_nodes
    
    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self._graph.edges)
    
    def add_edge(self, u: int, v: int, **attrs):
        """Add an edge with optional attributes."""
        self._graph.add_edge(u, v, **attrs)
    
    def add_node_attr(self, node: int, key: str, value: Any):
        """Add attribute to a node."""
        self._graph.add_node_attr(node, key, value)
    
    # Conversion methods
    def to_networkx(self, directed: bool = True):
        """Export to NetworkX graph."""
        return self._graph.to_networkx(directed=directed)
    
    def to_igraph(self, directed: bool = True):
        """Export to igraph graph."""
        return self._graph.to_igraph(directed=directed)
    
    def to_adjacency_matrix(self, weighted: bool = False, 
                           weight_attr: str = 'length') -> np.ndarray:
        """Export to numpy adjacency matrix."""
        return self._graph.to_adjacency_matrix(weighted=weighted, weight_attr=weight_attr)
    
    def to_sparse_matrix(self, weighted: bool = False, weight_attr: str = 'length'):
        """Export to scipy sparse matrix."""
        return self._graph.to_sparse_matrix(weighted=weighted, weight_attr=weight_attr)
    
    @classmethod
    def from_networkx(cls, G):
        """Create from NetworkX graph."""
        instance = cls(G.number_of_nodes())
        instance._graph = GraphRepresentation.from_networkx(G)
        return instance
    
    @classmethod
    def from_igraph(cls, g):
        """Create from igraph graph."""
        instance = cls(g.vcount())
        instance._graph = GraphRepresentation.from_igraph(g)
        return instance
    
    @classmethod
    def from_adjacency_matrix(cls, adj: np.ndarray, weighted: bool = False):
        """Create from adjacency matrix."""
        instance = cls(adj.shape[0])
        instance._graph = GraphRepresentation.from_adjacency_matrix(adj, weighted)
        return instance
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate that the graph satisfies structural constraints."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_nodes={self.n_nodes}, n_edges={self.n_edges})"
