[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22839827.svg)](https://doi.org/10.5281/zenodo.22839827)
[![Docs](https://readthedocs.org/projects/treegraphduals/badge/?version=latest)](https://treegraphduals.readthedocs.io/en/latest/)

# treegraphduals

This package is an extension of my Master's thesis: [The Horizontal Tunnelability Graph is Dual to Level Set Trees](http://hdl.handle.net/11714/10548).
I have tried to keep everything as compatible as possible with other popular time series, tree, and graph libraries.

This is an early release. The sections below separate what the package computes
today from what it is being built towards, so you can tell at a glance whether
the piece you need is ready.

## Install

```bash
pip install treegraphduals     # or: uv add treegraphduals
```

Requires Python 3.12+.

## Quickstart

```python
from treegraphduals.core import Tree
from treegraphduals.timeseries import TimeSeries

# Level-set tree of a time series
ts = TimeSeries.from_array([0, 3, 1, 4, 2, 5, 0])
tree = ts.to_level_set_tree()
tree  # Tree(n_nodes=6, n_leaves=3, root=0)
tree.horton_strahler_order_tree()  # 2
tree.max_horton_prunings()  # 3

# Level-set tree of a symbolic function, using exact critical points
import sympy as sp

t = sp.Symbol("t")
extrema = TimeSeries.from_function(
    (t - 1) * (t - 3) * (t - 5) * (t - 7), 1, 7, preserve_extrema=True
)
extrema.times  # [1.0, 1.764, 4.0, 6.236, 7.0]

# Hand the result to NetworkX or igraph
G = tree.to_networkx()  # DiGraph with 6 nodes and 5 edges
```

## What the package computes today

**Time series to trees**

- Level Set Trees of Time Series as Alternating Piecewise Linear Excursions with Slopes +/- 1 (Harris Paths)
- Level Set Trees of Time Series as Alternating Piecewise Linear Excursions of Arbitrary Slope
- Level Set Trees and Partial Trees of Arbitrary Time Series
- Level Set Trees of Arbitrary Functions via Sympy, using exact symbolic differentiation
- Harris Paths of Binary Trees (and Partial Trees)
- Local extrema detection, minimal excursions, and unit-slope excursions

**Trees**

- Horton-Strahler Orders of a Binary Tree, and of a tree as a whole
- Horton Pruning of a Tree, with series reduction, and the pruning count to eliminate a tree
- Plantedness, leaves, internal nodes, depths, subtree sizes, total length
- Traversals: depth-first, breadth-first, left-to-right, and edge contours
- Paths, distances, and ancestry queries between any two nodes
- Structures: `Tree`, `BinaryTree`, `Forest`, `DAG`, `Polytree`, `Graph`, `Multigraph`, `ErdosRenyi`, `GaltonWatsonTree`, `RealTree`

Trees may be weighted or unweighted.

**Interoperability** — every structure converts to and from

- NetworkX and igraph graphs
- NumPy adjacency matrices and SciPy sparse matrices
- Parent arrays, with left/right child information preserved for binary trees

This is the intended route to general graph-theoretic quantities for now:
export to NetworkX or igraph and use their algorithms.

**Visualization**

- Tree plots with disk (for duality), radial, force-directed, and hierarchical layouts
- Coloring by Horton-Strahler order, with node and edge annotations
- Time series, excursion, and extrema plots
- Trees drawn side by side with their Harris path

## Planned

None of the following is implemented yet. They are the roadmap, listed here
because they are where the package is going, not what it does now.

**Duals and visibility graphs**

- Duals of Binary Trees (Horizontal Tunnelability Graphs)
- Duals of Binary Trees w/ Partial Trees
- Dual of a Graph
- Horizontal Visibility Graphs of Piecewise Linear Excursions, Arbitrary Time Series, and Arbitrary Functions
- Horizon Visibility Graphs of Piecewise Linear Excursions, Arbitrary Time Series, and Arbitrary Functions
- Visibility Graphs of Piecewise Linear Excursions, Arbitrary Time Series, and Arbitrary Functions

**Further tree and time series constructions**

- Time Series Merge Trees of Arbitrary Time Series and of Arbitrary Functions
- Chiral Merge Trees of Arbitrary Time Series and of Arbitrary Functions
- U-shaped Segments of Arbitrary Time Series and of Arbitrary Functions
- Hurst Exponents
- Persistence Diagrams/Barcodes
- Time-varying trees, and hence time-varying duals, with appropriate interpretation on time series

**Native graph functionality**

The graph classes are intended to become general enough to allow for:

- Vertices with or without numerical weights as well as optional labels of arbitrary data type
- Directed or undirected Edges with or without numerical weights and optional labels of arbitrary data type
- Easy and efficient recall/computation of other graph representations and important quantities such as
- Adjacency Matrix, Incidence Matrix, Adjacency List, Laplacian Matrix, Graph Distance Matrix, etc.
- Graph Traversal Algorithms: DFS and BFS
- Find Shortest Path, Shortest Path Function, Hamiltonian Path, Topological Sort
- Find Graph Distances, Find Paths (Edge/Vertex Independent as well), Find Cycles (All, Eulerian, Hamiltonian, Postman, Shortest Tour)
- Check if two Graphs are Isomorphic
- Graph Union, Find Maximum Flow, Path Lengths, Mean Path Lengths, etc.
- Compute Graph Polynomials: Tutte, Chromatic, Flow
- Check if a Graph is a Subgraph of Another Graph
- Generate Neighborhood Graphs and Subgraphs from Graphs
- Get Connected Components of a Graph, k-Core Components, Weakly Connected Components
- Find Cliques
- Check if a Graph is a Tree (and if so how k-ary), if it is Acyclic, Bipartite, Planar, Loop Free, Simple, etc.
- Graph Self-Similarity, Small-World Property, Scale-Invariance
- Graph Metrics: Vertex and Edge Count, Vertex Degree/In and Out Degrees, Vertex Eccentricity
- Graph Radius, Graph Diameter, Graph Center, Graph Periphery, Vertex and Edge Connectivity
- Centrality Measures: Closeness, Betweenness, Edge Betweenness, Degree Centrality, Eigenvector Centrality, Katz Centrality, PageRank Centrality, HITS Centrality, Radiality, Status Centrality
- Reciprocity and Transitivity Measures: Graph Reciprocity & Global, Local, and Mean Clustering Coefficients
- Homophily, Assortative Mixing, and Similarity Measures: Assortativity, Vertex Correlation, Mean Neighbor Degree
- Mean Degree Connectivity, Vertex Dice Similarity, Vertex Jaccard Similarity, Vertex Cosine Similarity
- Mean Degree, Mean Path Length, Clustering Coefficients, Degree Distributions, Degree Sequences, many many other things

## Documentation

Full API documentation, tutorials, and examples: <https://treegraphduals.readthedocs.io>

## Citing

If you use this package, please cite it via the DOI above, or see
[`CITATION.cff`](CITATION.cff). The underlying thesis is
*The Horizontal Tunnelability Graph is Dual to Level Set Trees*
(University of Nevada, Reno, 2023).

## License

MIT
