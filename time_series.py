from typing import List, Tuple, Callable
import sympy as sp
import numpy as np

class TimeSeries:
    def __init__(self, data: List[float], times: List[float] = None):
        self.data = data
        self.times = times if times is not None else list(range(len(data)))

    def get_u_shaped_segments(self, method: str = 'simple') -> List[Tuple[int, int, int]]:
        """
        Find U-shaped segments in the time series.

        :param method: Method for finding U-shaped segments ('simple', 'adaptive', etc.)
        :return: List of (start, bottom, end) indices of U-shaped segments
        """
        pass

    def estimate_hurst_exponent(self, method: str = 'rs') -> float:
        """
        Estimate the Hurst exponent of the time series.

        :param method: Method for estimation ('rs' for rescaled range, 'dfa' for detrended fluctuation analysis, etc.)
        :return: Estimated Hurst exponent
        """
        pass

    def to_piecewise_linear(self, num_segments: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Convert the time series to a piecewise linear approximation.

        :param num_segments: Number of linear segments to use
        :return: Tuple of (breakpoints, slopes)
        """
        pass

    def to_sympy_function(self) -> sp.Expr:
        """
        Convert the time series to a SymPy function approximation.

        :return: SymPy expression representing the time series
        """
        pass

    def visualize(self, output_path: str = None):
        """
        Visualize the time series.

        :param output_path: Path to save the visualization (if None, display instead)
        """
        pass

    @staticmethod
    def generate_random_walk(length: int, hurst: float = 0.5) -> 'TimeSeries':
        """
        Generate a random walk time series with a given Hurst exponent.

        :param length: Length of the time series
        :param hurst: Hurst exponent (0.5 for Brownian motion)
        :return: A new TimeSeries object
        """
        pass

    @staticmethod
    def generate_fractional_brownian_motion(length: int, hurst: float) -> 'TimeSeries':
        """
        Generate a fractional Brownian motion time series.

        :param length: Length of the time series
        :param hurst: Hurst exponent
        :return: A new TimeSeries object
        """
        pass