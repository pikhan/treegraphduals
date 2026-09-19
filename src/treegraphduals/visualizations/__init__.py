"""
Visualization functions for trees, graphs, and time series.

Basic usage:
    from treegraphduals.visualizations import plot_tree, plot_timeseries

    # Plot tree
    fig, ax = plot_tree(tree, layout='disk')
    plt.show()

    # Plot time series
    fig, ax = plot_timeseries(times, values)
    plt.show()
"""

from .plot_combined import (
    plot_graph_overlay,
    plot_levelset_overlay,
    plot_tree_and_harris_path,
)
from .plot_timeseries import (
    plot_excursion,
    plot_timeseries,
    plot_timeseries_with_extrema,
)
from .plot_trees import (
    add_edge_annotation,
    add_node_annotation,
    color_by_horton_strahler,
    plot_tree,
)

__all__ = [
    "add_edge_annotation",
    "add_node_annotation",
    "color_by_horton_strahler",
    "plot_excursion",
    "plot_graph_overlay",
    "plot_levelset_overlay",
    "plot_timeseries",
    "plot_timeseries_with_extrema",
    "plot_tree",
    "plot_tree_and_harris_path",
]
