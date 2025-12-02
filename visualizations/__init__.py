# visualizations/__init__.py
"""
Visualization functions for trees, graphs, and time series.

Basic usage:
    from visualizations import plot_tree, plot_timeseries

    # Plot tree
    fig, ax = plot_tree(tree, layout='disk')
    plt.show()

    # Plot time series
    fig, ax = plot_timeseries(times, values)
    plt.show()
"""

from .plot_trees import (
    plot_tree,
    color_by_horton_strahler,
    add_node_annotation,
    add_edge_annotation
)

from .plot_timeseries import (
    plot_timeseries,
    plot_timeseries_with_extrema,
    plot_excursion
)

from .plot_combined import (
    plot_tree_and_harris_path,
    plot_levelset_overlay,
    plot_graph_overlay
)

__all__ = [
    # Trees
    'plot_tree',
    'color_by_horton_strahler',
    'add_node_annotation',
    'add_edge_annotation',
    # Time series
    'plot_timeseries',
    'plot_timeseries_with_extrema',
    'plot_excursion',
    # Combined
    'plot_tree_and_harris_path',
    'plot_levelset_overlay',
    'plot_graph_overlay',
]