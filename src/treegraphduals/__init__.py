"""Trees, their duals, graphs, and time series.

An extension of *The Horizontal Tunnelability Graph is Dual to Level Set Trees*
(Khan, University of Nevada, Reno, 2023).

The library is organized into subpackages, imported explicitly so that a bare
``import treegraphduals`` stays free of plotting dependencies:

- :mod:`treegraphduals.core` -- tree and graph data structures
- :mod:`treegraphduals.timeseries` -- time series analysis and tree conversions
- :mod:`treegraphduals.visualizations` -- plotting (requires matplotlib)

Examples
--------
>>> from treegraphduals.core import Tree
>>> tree = Tree(n_nodes=3, root=0)
>>> tree.add_edge(0, 1)
>>> tree.add_edge(0, 2)
>>> tree.n_edges
2
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("treegraphduals")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
