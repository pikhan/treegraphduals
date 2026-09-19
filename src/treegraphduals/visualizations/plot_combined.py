"""
Combined visualizations: trees with time series, overlays, etc.

TBD: Interactive features, level-set tree overlays, duality visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..core.tree import Tree
from .plot_trees import _compute_layout, _draw_disk_boundary, _draw_tree


def plot_tree_and_harris_path(
    tree: Tree,
    times: np.ndarray,
    values: np.ndarray,
    layout: str = "disk",
    figsize: tuple[float, float] = (16, 6),
    node_size: int = 300,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot tree and its Harris path side by side.

    Parameters
    ----------
    tree : Tree
        Tree to plot
    times : np.ndarray
        Harris path time points
    values : np.ndarray
        Harris path values
    layout : str
        Tree layout: 'disk', 'radial', 'force' or 'hierarchical'
    figsize : tuple
        Figure size
    node_size : int
        Size of tree nodes

    Returns
    -------
    fig : matplotlib.Figure
    axes : tuple of matplotlib.Axes
        (tree_ax, timeseries_ax)

    Examples
    --------
    >>> import numpy as np
    >>> from treegraphduals.core.tree import Tree
    >>> from treegraphduals.visualizations import plot_tree_and_harris_path
    >>> tree = Tree(n_nodes=3, root=0)
    >>> tree.add_edge(0, 1)
    >>> tree.add_edge(0, 2)
    >>> times, values = np.linspace(0, 1, 5), np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    >>> plt.close("all")
    >>> fig, (tree_ax, path_ax) = plot_tree_and_harris_path(tree, times, values)
    >>> len(plt.get_fignums())  # exactly one figure, nothing leaked
    1
    >>> plt.close(fig)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Draw the tree straight into ax1. Calling plot_tree() here would build a
    # second figure that pyplot keeps alive even after we drop the reference.
    pos = _compute_layout(tree, layout)
    if layout == "disk":
        _draw_disk_boundary(ax1)
    _draw_tree(
        tree,
        pos,
        ax1,
        show_node_labels=False,
        show_edge_lengths=False,
        node_size=node_size,
    )
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Tree", fontsize=14)

    # Plot Harris path on right
    ax2.plot(times, values, "b-", linewidth=1.5)
    ax2.fill_between(times, 0, values, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=2, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Height", fontsize=12)
    ax2.set_title("Harris Path", fontsize=14)

    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_levelset_overlay(
    times: np.ndarray,
    values: np.ndarray,
    tree: Tree,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot time series with level-set tree structure overlaid.

    TBD: Implementation requires mapping between tree nodes and time series extrema.

    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Time series values
    tree : Tree
        Level-set tree
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> # TBD: Add example once implemented
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(times, values, "b-", linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3)

    # TBD: Overlay tree structure
    # - Draw tree edges over time series
    # - Connect extrema according to tree parent-child relationships
    # - Requires careful mapping of tree nodes to time series indices

    ax.text(
        0.5,
        0.5,
        "TBD: Level-set tree overlay\nRequires extrema-to-node mapping",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Time Series with Level-Set Tree Overlay", fontsize=14)

    return fig, ax


def plot_graph_overlay(
    graphs: list,
    labels: list | None = None,
    colors: list | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple graphs overlaid on same axes.

    TBD: For comparing different graph structures (original vs dual, etc.)

    Parameters
    ----------
    graphs : list of Tree
        Graphs to overlay
    labels : list of str, optional
        Labels for each graph
    colors : list of str, optional
        Colors for each graph
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> # TBD: Add example
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.text(
        0.5,
        0.5,
        "TBD: Graph overlay visualization\nFor dual graphs, etc.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )

    return fig, ax


# TBD: Interactive visualizations with plotly
# def plot_interactive_tree(tree: Tree, ...):
#     """Interactive tree plot with hover info, click handlers, etc."""
#     pass

# TBD: Persistence diagrams, barcodes (for TDA)
# def plot_persistence_diagram(...):
#     """Plot persistence diagram for TDA."""
#     pass

# TBD: Visibility graph visualizations
# def plot_visibility_graph(...):
#     """Plot horizontal/horizon visibility graph."""
#     pass
