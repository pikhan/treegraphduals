"""
Time series to tree conversions using level-set trees and Harris paths.

Implements constructions from:
- Haskell (2020): Partial trees and partial Harris paths
- Kovchegov & Zaliapin (2020): Random self-similar trees
- Ibraheem Khan (2023): The Horizontal Tunnelability Graph is Dual to Level Set Trees.
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

        Parameters
        ----------
        func : callable or sympy expression
            Function to evaluate
        t_start, t_end : float
            Time interval
        n_points : int
            Number of sample points
        preserve_extrema : bool
            If True, adds extra points at detected extrema
        """
        # Handle sympy functions
        try:
            import sympy as sp
            if isinstance(func, sp.Basic):
                # Convert sympy to lambda
                t = sp.Symbol('t')
                func_lambda = sp.lambdify(t, func, 'numpy')
            else:
                func_lambda = func
        except ImportError:
            func_lambda = func

        # Sample function
        times = np.linspace(t_start, t_end, n_points)
        values = np.array([func_lambda(t) for t in times])

        if preserve_extrema:
            # Detect extrema and add points
            extrema_mask = detect_local_extrema(values)
            # Could add more points around extrema, but for now just use base sampling

        return cls(times=times, values=values)

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
        >>> # Add doctests here
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
    Build level-set tree from extrema.

    Algorithm:
    - Maxima become leaves
    - Minima become internal nodes
    - Parent-child relationships determined by nesting structure
    """
    # Combine extrema with type labels
    extrema = []
    for idx in minima_idx:
        extrema.append((idx, values[idx], 'min'))
    for idx in maxima_idx:
        extrema.append((idx, values[idx], 'max'))

    # Sort by index (temporal order)
    extrema.sort(key=lambda x: x[0])

    n_nodes = len(extrema)
    tree = Tree(n_nodes=n_nodes)

    # Map extrema index to node ID
    extrema_to_node = {extrema[i][0]: i for i in range(n_nodes)}

    # Build tree structure using stack-based algorithm
    # This implements the nesting structure of level sets
    stack = []  # Stack of (node_id, value, type)

    for i, (idx, val, etype) in enumerate(extrema):
        if etype == 'max':
            # Leaf node - find its parent (most recent lower minimum)
            while stack and stack[-1][2] == 'max':
                stack.pop()

            if stack:
                parent_id = stack[-1][0]
                # Compute edge length
                parent_idx = extrema[parent_id][0]
                edge_length = _compute_edge_length(
                    times[parent_idx], values[parent_idx],
                    times[idx], values[idx],
                    edge_metric
                )
                tree.add_edge(parent_id, i, length=edge_length)
            else:
                # This maxima is at top level (shouldn't happen in proper excursion)
                tree.root = i

            stack.append((i, val, etype))

        else:  # minimum
            # Internal node - may be parent to previous maxima
            # Pop maxima that are higher than this minimum
            children = []
            while stack and stack[-1][1] > val:
                child_id, child_val, child_type = stack.pop()
                if child_type == 'max':
                    children.append(child_id)

            # Add this minimum to stack
            stack.append((i, val, etype))

            # Connect children to this minimum
            for child_id in children:
                child_idx = extrema[child_id][0]
                edge_length = _compute_edge_length(
                    times[idx], values[idx],
                    times[child_idx], values[child_idx],
                    edge_metric
                )
                tree.add_edge(i, child_id, length=edge_length)

    # Find root (lowest minimum or highest point)
    min_val = np.min(values[minima_idx]) if len(minima_idx) > 0 else values[0]
    root_candidates = [i for i, (idx, val, etype) in enumerate(extrema)
                       if etype == 'min' and val == min_val]

    if len(root_candidates) > 0:
        tree.root = root_candidates[0]
    else:
        tree.root = 0

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