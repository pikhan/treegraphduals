"""
Time series to tree conversions using level-set trees and Harris paths.

Implements constructions from:
- Haskell (2020): Partial trees and partial Harris paths
- Kovchegov & Zaliapin (2020): Random self-similar trees
- Khan (2023): The Horizontal Tunnelability Graph is Dual to Level Set Trees.
"""

import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any
from collections import deque
import sys
import os

from core.tree import Tree


class TimeSeries:
    """
    Time series representation for level-set tree construction.

    Handles conversion between time series and trees via level-set trees
    and Harris path correspondences (Haskell 2020, Kovchegov & Zaliapin 2020).

    Attributes
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Function values at time points
    is_excursion : bool
        Whether series starts and ends at same value
    """

    def __init__(self, times: Optional[np.ndarray] = None,
                 values: Optional[np.ndarray] = None):
        """
        Initialize time series.

        Parameters
        ----------
        times : array-like, optional
            Time points. If None, uses indices.
        values : array-like
            Function values
        """
        if values is None:
            raise ValueError("Values must be provided")

        self.values = np.asarray(values, dtype=float)

        if times is None:
            self.times = np.arange(len(self.values), dtype=float)
        else:
            self.times = np.asarray(times, dtype=float)

        if len(self.times) != len(self.values):
            raise ValueError("Times and values must have same length")

        self.is_excursion = np.abs(self.values[0] - self.values[-1]) < 1e-10

    @classmethod
    def from_array(cls, data: Union[np.ndarray, List]) -> 'TimeSeries':
        """Create from numpy array or list (values only)."""
        return cls(values=data)

    @classmethod
    def from_pandas(cls, series_or_df) -> 'TimeSeries':
        """
        Create from pandas Series or DataFrame.

        Parameters
        ----------
        series_or_df : pd.Series or pd.DataFrame
            If DataFrame, uses first column
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for this method")

        if isinstance(series_or_df, pd.Series):
            times = series_or_df.index.values if hasattr(series_or_df.index, 'values') else np.arange(len(series_or_df))
            values = series_or_df.values
        elif isinstance(series_or_df, pd.DataFrame):
            times = series_or_df.index.values if hasattr(series_or_df.index, 'values') else np.arange(len(series_or_df))
            values = series_or_df.iloc[:, 0].values
        else:
            raise TypeError("Input must be pandas Series or DataFrame")

        return cls(times=times, values=values)

    @classmethod
    def from_function(cls, func, t_start: float = 0, t_end: float = 1,
                      n_points: int = 100, preserve_extrema: bool = True) -> 'TimeSeries':
        """
        Create from continuous function (sympy or callable).

        If preserve_extrema=True, only includes t_start, t_end, and local extrema points.
        Otherwise, uses n_points uniform sampling.

        Parameters
        ----------
        func : callable or sympy expression
            Function to evaluate
        t_start, t_end : float
            Time interval
        n_points : int
            Number of sample points (only used if preserve_extrema=False or for initial detection)
        preserve_extrema : bool
            If True, returns time series with only extrema points (piecewise linear)
            If False, returns uniformly sampled time series

        Returns
        -------
        TimeSeries
            Time series with specified points

        Examples
        --------
        .. plot::
           :include-source:
           :context: close-figs

           from timeseries import TimeSeries
           from visualizations import plot_timeseries
           from sympy import symbols
           import matplotlib.pyplot as plt
           t = symbols('t')
           myPolynomial = (t-1)*(t-3)*(t-5)*(t-7)
           myTS = TimeSeries.from_function(myPolynomial, 1, 7, preserve_extrema=False)
           fig, ax = plot_timeseries(myTS.times, myTS.values, title="Original Function")
           plt.show()

        .. plot::
           :context: close-figs

           # With extrema only (piecewise linear)
           myTS_extrema = TimeSeries.from_function(myPolynomial, 1, 7, preserve_extrema=True)
           fig, ax = plot_timeseries(myTS_extrema.times, myTS_extrema.values, title="extrema")
           plt.show()

        .. plot::
           :context: close-figs

           myTS_extrema_minimal = myTS_extrema.to_minimal_excursion()
           fig, ax = plot_timeseries(myTS_extrema_minimal.times, myTS_extrema_minimal.values, title="minimal")
           plt.show()
        """
        # Handle sympy functions
        try:
            import sympy as sp
            if isinstance(func, sp.Basic):
                # Convert sympy to lambda
                t = sp.Symbol('t')
                func_lambda = sp.lambdify(t, func, 'numpy')
                is_sympy = True
            else:
                func_lambda = func
                is_sympy = False
        except ImportError:
            func_lambda = func
            is_sympy = False

        if not preserve_extrema:
            # Simple uniform sampling
            times = np.linspace(t_start, t_end, n_points)
            values = np.array([func_lambda(t) for t in times])
            return cls(times=times, values=values)

        # preserve_extrema=True: only keep t_start, t_end, and local extrema

        # For sympy functions, use exact symbolic differentiation
        if is_sympy:
            try:
                import sympy as sp
                t_sym = sp.Symbol('t')

                # Compute derivative
                func_derivative = sp.diff(func, t_sym)

                # Find critical points
                critical_points = sp.solve(func_derivative, t_sym)

                # Filter to real numbers in interval
                extrema_times = [t_start]

                for cp in critical_points:
                    try:
                        cp_float = float(cp.evalf())
                        if t_start < cp_float < t_end:  # Strictly interior
                            extrema_times.append(cp_float)
                    except:
                        continue

                extrema_times.append(t_end)
                extrema_times = sorted(extrema_times)

                # Evaluate function at extrema
                extrema_values = np.array([func_lambda(t) for t in extrema_times])

                return cls(times=np.array(extrema_times), values=extrema_values)

            except Exception as e:
                # Fall back to numerical method if symbolic fails
                print(f"Symbolic differentiation failed: {e}, using numerical method")
                pass

        # Numerical method for non-sympy or if symbolic failed
        # Use fine sampling with better extrema detection
        times_dense = np.linspace(t_start, t_end, max(n_points, 1000))
        values_dense = np.array([func_lambda(t) for t in times_dense])

        # Use scipy to find peaks (more robust than manual detection)
        try:
            from scipy.signal import find_peaks

            # Find maxima
            maxima_idx, _ = find_peaks(values_dense)

            # Find minima (peaks of negative function)
            minima_idx, _ = find_peaks(-values_dense)

            # Combine
            extrema_indices = sorted(list(maxima_idx) + list(minima_idx))

            # Add endpoints
            extrema_indices = [0] + extrema_indices + [len(times_dense) - 1]
            extrema_indices = sorted(set(extrema_indices))

        except ImportError:
            # Fallback: Manual detection with tolerance
            extrema_indices = [0]  # Start

            # Use a wider window for comparison to handle flat regions
            window = max(3, len(times_dense) // 100)

            for i in range(window, len(values_dense) - window):
                # Check if local maximum (within window)
                is_max = all(values_dense[i] >= values_dense[i - j] for j in range(1, window + 1))
                is_max = is_max and all(values_dense[i] >= values_dense[i + j] for j in range(1, window + 1))

                # Check if local minimum (within window)
                is_min = all(values_dense[i] <= values_dense[i - j] for j in range(1, window + 1))
                is_min = is_min and all(values_dense[i] <= values_dense[i + j] for j in range(1, window + 1))

                # Check if it's actually an extremum (not flat everywhere)
                left_change = abs(values_dense[i] - values_dense[i - window])
                right_change = abs(values_dense[i] - values_dense[i + window])

                if (is_max or is_min) and (left_change > 1e-10 or right_change > 1e-10):
                    # Avoid duplicates (don't add if very close to previous extremum)
                    if not extrema_indices or i - extrema_indices[-1] > window // 2:
                        extrema_indices.append(i)

            extrema_indices.append(len(times_dense) - 1)  # End

        # Extract extrema points
        times_extrema = times_dense[extrema_indices]
        values_extrema = values_dense[extrema_indices]

        return cls(times=times_extrema, values=values_extrema)

    @classmethod
    def harris_path_from_function(cls, func, t_start: float = 0, t_end: float = 1,
                                  n_sample: int = 1000) -> 'TimeSeries':
        """
        Create Harris path directly from function.

        Builds level-set tree from function, then constructs Harris path
        with alternating ±1 slopes.

        Parameters
        ----------
        func : callable or sympy expression
            Function to convert
        t_start, t_end : float
            Time interval
        n_sample : int
            Sampling density for extrema detection

        Returns
        -------
        TimeSeries
            Harris path with ±1 slopes

        Examples
        --------
        .. plot::
           :include-source:
           :context: close-figs

           from matplotlib import pyplot as plt
           from visualizations import plot_timeseries
           from sympy import symbols
           t = symbols('t')
           poly = (t-1)*(t-3)*(t-5)*(t-7)
           harris = TimeSeries.harris_path_from_function(poly, 1, 7)
           ax, fig = plot_timeseries(harris.times, harris.values)
           plt.show()
        """
        # Step 1: Sample function to get extrema
        ts_sampled = cls.from_function(func, t_start, t_end,
                                       n_points=n_sample, preserve_extrema=True)

        # Step 2: Build level-set tree
        tree = ts_sampled.to_level_set_tree(edge_metric='vertical', unit_slopes=False, return_partial=True)

        # Step 3: Construct Harris path from tree
        harris_path = tree_to_harris_path(tree)

        return harris_path

    def find_local_extrema(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find indices of local minima and maxima.

        Returns
        -------
        minima_indices : np.ndarray
            Indices of local minima
        maxima_indices : np.ndarray
            Indices of local maxima
        """
        minima = []
        maxima = []

        n = len(self.values)

        # Check interior points
        for i in range(1, n - 1):
            if self.values[i] < self.values[i - 1] and self.values[i] < self.values[i + 1]:
                minima.append(i)
            elif self.values[i] > self.values[i - 1] and self.values[i] > self.values[i + 1]:
                maxima.append(i)

        # Check endpoints
        if n > 1:
            if self.values[0] < self.values[1]:
                minima.insert(0, 0)
            elif self.values[0] > self.values[1]:
                maxima.insert(0, 0)

            if self.values[-1] < self.values[-2]:
                minima.append(n - 1)
            elif self.values[-1] > self.values[-2]:
                maxima.append(n - 1)

        return np.array(minima, dtype=int), np.array(maxima, dtype=int)

    def to_unit_slope_excursion(self) -> 'TimeSeries':
        """
        Convert to piecewise linear excursion with alternating ±1 slopes.

        Preserves all local extrema, creates proper excursion starting
        and ending at 0 with slopes of exactly ±1.

        Returns
        -------
        TimeSeries
            Excursion with unit slopes
        """
        minima_idx, maxima_idx = self.find_local_extrema()

        # Combine and sort extrema
        extrema_idx = np.sort(np.concatenate([minima_idx, maxima_idx]))
        extrema_vals = self.values[extrema_idx]

        if len(extrema_idx) == 0:
            # Constant function
            return TimeSeries(times=np.array([0.0, 1.0]), values=np.array([0.0, 0.0]))

        # Shift so minimum is at 0
        min_val = np.min(extrema_vals)
        shifted_vals = extrema_vals - min_val

        # Build excursion with ±1 slopes
        new_times = [0.0]
        new_values = [0.0]

        current_time = 0.0

        # Add initial segment from 0 to first extremum
        if shifted_vals[0] > 0:
            current_time += shifted_vals[0]
            new_times.append(current_time)
            new_values.append(shifted_vals[0])

        # Add segments between extrema
        for i in range(len(shifted_vals) - 1):
            v1, v2 = shifted_vals[i], shifted_vals[i + 1]
            delta = abs(v2 - v1)
            current_time += delta
            new_times.append(current_time)
            new_values.append(v2)

        # Add final segment back to 0
        if shifted_vals[-1] > 0:
            current_time += shifted_vals[-1]
            new_times.append(current_time)
            new_values.append(0.0)

        return TimeSeries(times=np.array(new_times), values=np.array(new_values))

    def to_minimal_excursion(self) -> 'TimeSeries':
        """
        Convert to minimal excursion (Haskell 2020, Definition 3.2.3).

        Extends endpoints downward with slope ±1 until they meet at
        min(values), creating smallest excursion containing the series.

        Returns
        -------
        TimeSeries
            Minimal excursion
        """
        min_val = np.min(self.values)

        # Extend start downward
        start_val = self.values[0]
        start_extension = start_val - min_val

        # Extend end downward
        end_val = self.values[-1]
        end_extension = end_val - min_val

        # Build extended series
        n_start = int(np.ceil(start_extension)) + 1
        n_end = int(np.ceil(end_extension)) + 1

        start_times = np.linspace(-start_extension, 0, n_start)[:-1]
        start_values = np.linspace(0, start_val, n_start)[:-1]

        # Shift original series
        shifted_times = self.times - self.times[0]
        shifted_values = self.values - min_val

        end_times = shifted_times[-1] + np.linspace(0, end_extension, n_end)[1:]
        end_values = np.linspace(end_val - min_val, 0, n_end)[1:]

        full_times = np.concatenate([start_times, shifted_times, end_times])
        full_values = np.concatenate([start_values, shifted_values, end_values])

        return TimeSeries(times=full_times, values=full_values)

    def to_level_set_tree(self,
                          edge_metric: str = 'euclidean',
                          unit_slopes: bool = False,
                          force_excursion: bool = True,
                          return_partial: bool = False) -> Union[Tree, Tuple[Tree, int, int]]:
        """
        Convert time series to level-set tree.

        Parameters
        ----------
        edge_metric : str
            Edge length computation:
            - 'euclidean': sqrt((Δt)² + (Δy)²) [DEFAULT]
            - 'vertical': |Δy| (vertical distance only)
            - 'manhattan': |Δt| + |Δy|
            - 'temporal': |Δt| (temporal distance only)
        unit_slopes : bool
            If True, converts to unit slope (±1) excursion first
        force_excursion : bool
            If True and not an excursion, creates minimal excursion
        return_partial : bool
            If True, returns (tree, start_idx, end_idx) marking partial tree bounds

        Returns
        -------
        tree : Tree
            Level-set tree with edge lengths
        start_idx, end_idx : int (if return_partial=True)
            Indices marking the partial tree region

        Examples
        --------
        .. plot::
           :include-source:
           :context: close-figs

           from matplotlib import pyplot as plt
           from visualizations import plot_timeseries, plot_tree
           from sympy import symbols
           t = symbols('t')
           poly = (t-1)*(t-3)*(t-5)*(t-7)
           harris = TimeSeries.harris_path_from_function(poly, 1, 7)
           fig, ax = plot_timeseries(harris.times, harris.values)
           plt.show()

        .. plot::
           :context: close-figs

           myTree = harris.to_level_set_tree()
           ax, fig = plot_tree(myTree, layout='disk')
           plt.show()

        .. plot::
           :context: close-figs

           times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
           values = np.array([0.0, 6.0, 1.0, 3.0, 2.0, 4.0, 1.5, 5.0, 0.0])
           ts_nested = TimeSeries(times=times, values=values)
           tree_nested = ts_nested.to_level_set_tree(edge_metric='vertical')
           fig, ax = plot_timeseries(times, values, title="A Few Nested Peaks")
           plt.show()

        .. plot::
           :context: close-figs

           fig, ax = plot_tree(tree_nested, layout='disk', show_node_labels=True)
           plt.show()

        .. plot::
           :context: close-figs

           times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
           values = np.array([0.0, 2.0, 1.0, 3.0, 1.0, 2.0, 0.0])
           ts_triangle = TimeSeries(times=times, values=values)
           tree_triangle = ts_triangle.to_level_set_tree(edge_metric='vertical')
           fig, ax = plot_timeseries(times, values, title="Triangle Wave")
           plt.show()

        .. plot::
           :context: close-figs

           fig, ax = plot_tree(tree_triangle, layout='disk', show_node_labels=True)
           plt.show()
        """
        # Preprocess if needed
        ts = self
        start_idx = 0
        end_idx = len(self.values) - 1

        if unit_slopes:
            ts = self.to_unit_slope_excursion()
        elif force_excursion and not self.is_excursion:
            ts = self.to_minimal_excursion()
            # Track where original series starts/ends in extended version
            # This is approximate - more precise tracking would require careful indexing
            start_idx = np.searchsorted(ts.times, self.times[0])
            end_idx = np.searchsorted(ts.times, self.times[-1])

        # Find extrema
        minima_idx, maxima_idx = ts.find_local_extrema()

        if len(maxima_idx) == 0:
            # No peaks - single node tree
            tree = Tree(n_nodes=1, root=0)
            if return_partial:
                return tree, start_idx, end_idx
            return tree

        # Build level-set tree structure
        tree = _build_level_set_tree_structure(
            ts.times, ts.values, minima_idx, maxima_idx, edge_metric
        )

        if return_partial:
            return tree, start_idx, end_idx
        return tree


def _build_level_set_tree_structure(times: np.ndarray,
                                    values: np.ndarray,
                                    minima_idx: np.ndarray,
                                    maxima_idx: np.ndarray,
                                    edge_metric: str) -> Tree:
    """
    Build level-set tree from excursion using left-to-right connection.

    Algorithm from level-set tree definition:
    - Leaves at local maxima
    - Internal nodes at local minima
    - Root at baseline (value=0)
    - Connect vertices left-to-right
    """

    # Verify excursion
    if not (np.abs(values[0] - values[-1]) < 1e-10):
        raise ValueError("Must be an excursion (start and end at same value)")

    # Get interior extrema (exclude endpoints)
    interior_minima = [idx for idx in minima_idx if 0 < idx < len(values) - 1]
    interior_maxima = [idx for idx in maxima_idx if 0 < idx < len(values) - 1]

    # Build list of all nodes
    nodes = [(0, values[0], 'root')]
    for idx in interior_minima:
        nodes.append((idx, values[idx], 'min'))
    for idx in interior_maxima:
        nodes.append((idx, values[idx], 'max'))

    # Sort by time (left-to-right)
    nodes.sort(key=lambda x: x[0])

    n_nodes = len(nodes)
    tree = Tree(n_nodes=n_nodes)
    tree.root = 0

    # Map time index to node id
    idx_to_node = {nodes[i][0]: i for i in range(n_nodes)}

    # Separate valleys and peaks in order
    valleys = [(i, nodes[i][0], nodes[i][1]) for i in range(n_nodes) if nodes[i][2] == 'min']
    peaks = [(i, nodes[i][0], nodes[i][1]) for i in range(n_nodes) if nodes[i][2] == 'max']

    # Connect root to first valley
    if valleys:
        first_valley_node = valleys[0][0]
        first_valley_idx = valleys[0][1]

        edge_length = _compute_edge_length(
            times[0], values[0],
            times[first_valley_idx], values[first_valley_idx],
            edge_metric
        )
        tree.add_edge(tree.root, first_valley_node, length=edge_length)

    # Connect each valley to its left and right children
    for i, (valley_node, valley_idx, valley_val) in enumerate(valleys):
        # LEFT child: peak immediately before this valley
        left_peak = None
        for peak_node, peak_idx, peak_val in peaks:
            if peak_idx < valley_idx:
                if left_peak is None or peak_idx > left_peak[1]:
                    left_peak = (peak_node, peak_idx, peak_val)

        if left_peak:
            peak_node, peak_idx, peak_val = left_peak
            edge_length = _compute_edge_length(
                times[valley_idx], values[valley_idx],
                times[peak_idx], values[peak_idx],
                edge_metric
            )
            tree.add_edge(valley_node, peak_node, length=edge_length)
            # Remove this peak so it's not used again
            peaks = [(n, idx, v) for (n, idx, v) in peaks if idx != peak_idx]

        # RIGHT child: next valley OR next peak (if last valley)
        if i < len(valleys) - 1:
            # Not last valley → RIGHT child is next valley
            next_valley_node, next_valley_idx, next_valley_val = valleys[i + 1]
            edge_length = _compute_edge_length(
                times[valley_idx], values[valley_idx],
                times[next_valley_idx], values[next_valley_idx],
                edge_metric
            )
            tree.add_edge(valley_node, next_valley_node, length=edge_length)
        else:
            # Last valley → RIGHT child is next peak (if any)
            right_peak = None
            for peak_node, peak_idx, peak_val in peaks:
                if peak_idx > valley_idx:
                    if right_peak is None or peak_idx < right_peak[1]:
                        right_peak = (peak_node, peak_idx, peak_val)

            if right_peak:
                peak_node, peak_idx, peak_val = right_peak
                edge_length = _compute_edge_length(
                    times[valley_idx], values[valley_idx],
                    times[peak_idx], values[peak_idx],
                    edge_metric
                )
                tree.add_edge(valley_node, peak_node, length=edge_length)

    return tree


def _compute_edge_length(t1: float, y1: float,
                         t2: float, y2: float,
                         metric: str) -> float:
    """
    Compute edge length between two points.

    Parameters
    ----------
    t1, y1 : float
        First point (time, value)
    t2, y2 : float
        Second point (time, value)
    metric : str
        Distance metric

    Returns
    -------
    float
        Edge length
    """
    if metric == 'euclidean':
        return np.sqrt((t2 - t1) ** 2 + (y2 - y1) ** 2)
    elif metric == 'vertical':
        return abs(y2 - y1)
    elif metric == 'manhattan':
        return abs(t2 - t1) + abs(y2 - y1)
    elif metric == 'temporal':
        return abs(t2 - t1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def tree_to_harris_path(tree: Tree,
                        start_node: Optional[int] = None,
                        end_node: Optional[int] = None) -> TimeSeries:
    """
    Construct Harris path from tree (tree → time series).

    Creates piecewise linear function with slopes determined by edge lengths.
    For weighted trees, slopes = edge_length / unit_time.

    Parameters
    ----------
    tree : Tree
        Input tree (can be partial tree)
    start_node : int, optional
        Starting node for partial Harris path. If None, uses root.
    end_node : int, optional
        Ending node for partial Harris path. If None, uses full tree.

    Returns
    -------
    TimeSeries
        Harris path (excursion if full tree, partial path otherwise)

    Examples
    --------
    >>> # Add doctests here
    """
    if start_node is None:
        if isinstance(tree, tuple):
            tree = tree[0]
        start_node = tree.root

    # Perform tree traversal (DFS contour)
    times = [0.0]
    values = [0.0]
    current_time = 0.0
    current_height = 0.0

    visited = set()

    def traverse_subtree(node: int, height: float, time: float):
        """DFS traversal creating contour path."""
        nonlocal current_time, current_height

        visited.add(node)
        children = tree.get_children(node)

        for child in children:
            if child not in visited:
                # Descend to child (upward in path)
                edge_length = tree.get_edge_length(node, child)
                current_time += edge_length
                current_height += edge_length

                times.append(current_time)
                values.append(current_height)

                # Traverse child subtree
                traverse_subtree(child, current_height, current_time)

                # Return from child (downward in path)
                current_time += edge_length
                current_height -= edge_length

                times.append(current_time)
                values.append(current_height)

    traverse_subtree(start_node, 0.0, 0.0)

    return TimeSeries(times=np.array(times), values=np.array(values))


def detect_local_extrema(values: np.ndarray) -> np.ndarray:
    """
    Detect local extrema (both minima and maxima).

    Parameters
    ----------
    values : np.ndarray
        Array of values

    Returns
    -------
    np.ndarray
        Boolean mask of extrema locations
    """
    n = len(values)
    if n < 3:
        return np.zeros(n, dtype=bool)

    extrema = np.zeros(n, dtype=bool)

    for i in range(1, n - 1):
        if (values[i] > values[i - 1] and values[i] > values[i + 1]) or \
                (values[i] < values[i - 1] and values[i] < values[i + 1]):
            extrema[i] = True

    return extrema


# Convenience functions

def timeseries_to_tree(data: Any, **kwargs) -> Tree:
    """
    Convert any time series format to level-set tree.

    Parameters
    ----------
    data : array-like, pd.Series, pd.DataFrame, callable
        Time series data in any supported format
    **kwargs
        Passed to TimeSeries.to_level_set_tree()

    Returns
    -------
    Tree
        Level-set tree

    Examples
    --------
    >>> # Add doctests here
    """
    # Auto-detect format
    if callable(data):
        ts = TimeSeries.from_function(data)
    elif hasattr(data, 'values'):  # pandas
        ts = TimeSeries.from_pandas(data)
    else:  # array-like
        ts = TimeSeries.from_array(data)

    return ts.to_level_set_tree(**kwargs)


def tree_to_timeseries(tree: Tree, **kwargs) -> TimeSeries:
    """
    Convert tree to Harris path time series.

    Parameters
    ----------
    tree : Tree
        Input tree
    **kwargs
        Passed to tree_to_harris_path()

    Returns
    -------
    TimeSeries
        Harris path

    Examples
    --------
    >>> # Add doctests here
    """
    return tree_to_harris_path(tree, **kwargs)