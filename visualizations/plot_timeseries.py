# visualizations/plot_timeseries.py
"""
Time series visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_timeseries(times: np.ndarray,
                    values: np.ndarray,
                    figsize: Tuple[float, float] = (12, 4),
                    title: Optional[str] = None,
                    xlabel: str = 'Time',
                    ylabel: str = 'Value',
                    grid: bool = True,
                    **plot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time series.

    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Values at time points
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    grid : bool
        Show grid (default: True)
    **plot_kwargs
        Passed to plt.plot()

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> times = np.linspace(0, 10, 100)
    >>> values = np.sin(times)
    >>> fig, ax = plot_timeseries(times, values, title="Sine wave")
    >>> # plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Default plot style
    default_kwargs = {'linewidth': 1.5, 'color': 'blue'}
    default_kwargs.update(plot_kwargs)

    ax.plot(times, values, **default_kwargs)

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if grid:
        ax.grid(True, alpha=0.3)

    return fig, ax


def plot_timeseries_with_extrema(times: np.ndarray,
                                 values: np.ndarray,
                                 mark_extrema: bool = True,
                                 **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time series with local extrema marked.

    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Values
    mark_extrema : bool
        Mark local minima and maxima
    **kwargs
        Passed to plot_timeseries()

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> times = np.linspace(0, 10, 100)
    >>> values = np.sin(times) + 0.1 * np.sin(5 * times)
    >>> fig, ax = plot_timeseries_with_extrema(times, values)
    >>> # plt.show()
    """
    fig, ax = plot_timeseries(times, values, **kwargs)

    if mark_extrema:
        # Find extrema
        minima_idx, maxima_idx = _find_extrema(values)

        # Mark minima
        if len(minima_idx) > 0:
            ax.scatter(times[minima_idx], values[minima_idx],
                       c='red', s=100, marker='v', zorder=3,
                       label='Local minima')

        # Mark maxima
        if len(maxima_idx) > 0:
            ax.scatter(times[maxima_idx], values[maxima_idx],
                       c='green', s=100, marker='^', zorder=3,
                       label='Local maxima')

        ax.legend()

    return fig, ax


def _find_extrema(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find indices of local minima and maxima."""
    minima = []
    maxima = []

    n = len(values)
    for i in range(1, n - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)
        elif values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)

    return np.array(minima), np.array(maxima)


def plot_excursion(times: np.ndarray,
                   values: np.ndarray,
                   **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot excursion (time series starting and ending at 0).

    Highlights the zero line and fills area under curve.

    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Values (should start and end at 0)
    **kwargs
        Passed to plot_timeseries()

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes

    Examples
    --------
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> values = np.array([0, 1, 2, 1, 0])
    >>> fig, ax = plot_excursion(times, values)
    >>> # plt.show()
    """
    fig, ax = plot_timeseries(times, values, **kwargs)

    # Fill area under curve
    ax.fill_between(times, 0, values, alpha=0.3, color='blue')

    # Emphasize zero line
    ax.axhline(y=0, color='black', linewidth=2, linestyle='-', alpha=0.5)

    return fig, ax