# timeseries/__init__.py
"""
Time series to tree conversions using level-set trees and Harris paths.

Basic usage:
    from timeseries import TimeSeries, timeseries_to_tree, tree_to_timeseries

    # Create time series from various formats
    ts = TimeSeries.from_array([1, 2, 3, 2, 1])
    ts = TimeSeries.from_pandas(df)
    ts = TimeSeries.from_function(lambda t: np.sin(t), 0, 10)

    # Convert to tree
    tree = ts.to_level_set_tree(edge_metric='euclidean')

    # Or use convenience function
    tree = timeseries_to_tree([1, 2, 3, 2, 1])

    # Convert tree to Harris path
    harris_path = tree_to_timeseries(tree)
"""

from .timeseries import (
    TimeSeries,
    timeseries_to_tree,
    tree_to_timeseries,
    tree_to_harris_path,
    detect_local_extrema,
)

__all__ = [
    # Main class
    'TimeSeries',
    # Conversion functions
    'timeseries_to_tree',
    'tree_to_timeseries',
    'tree_to_harris_path',
    # Utilities
    'detect_local_extrema',
]