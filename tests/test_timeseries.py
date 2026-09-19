"""
Test suite for the TimeSeries class.

Run with:
    pytest tests/test_timeseries.py
    pytest tests/test_timeseries.py -v
    pytest tests/test_timeseries.py --cov=treegraphduals.timeseries
"""

import numpy as np
import pytest
import sympy as sp

from treegraphduals.timeseries import TimeSeries

t = sp.Symbol("t")


class TestSymbolicExtrema:
    """Critical-point finding on the symbolic (sympy) path of from_function."""

    @staticmethod
    def interior_times(expr, t_start, t_end):
        """Return the interior extrema times found for expr, endpoints dropped."""
        ts = TimeSeries.from_function(expr, t_start, t_end, preserve_extrema=True)
        assert ts.times[0] == pytest.approx(t_start)
        assert ts.times[-1] == pytest.approx(t_end)
        assert list(ts.times) == sorted(ts.times)
        return list(ts.times[1:-1])

    def test_polynomial_extrema(self):
        """A quartic's three critical points are found exactly."""
        found = self.interior_times((t - 1) * (t - 3) * (t - 5) * (t - 7), 1, 7)
        assert found == pytest.approx([4 - np.sqrt(5), 4.0, 4 + np.sqrt(5)])

    def test_periodic_function_beyond_principal_solutions(self):
        """sin(t) on [0, 10] has three interior extrema, not just the first two.

        sympy.solve(cos(t), t) returns only the principal solutions pi/2 and
        3*pi/2, which would silently drop the maximum at 5*pi/2.
        """
        found = self.interior_times(sp.sin(t), 0, 10)
        assert found == pytest.approx([np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2])

    def test_real_roots_returned_in_complex_form(self):
        """Cubic roots written via Cardano's formula are still recognized as real.

        sympy returns the three real roots of t**3 - 4*t + 1/3 in terms of I,
        with imaginary parts around 1e-23; float() raises on all three.
        """
        found = self.interior_times(t**4 / 4 - 2 * t**2 + t / 3, -3, 3)
        assert len(found) == 3
        assert found == pytest.approx([-2.040432, 0.083479, 1.956954], abs=1e-5)

    def test_no_real_critical_points(self):
        """A function whose derivative has only complex roots keeps just the endpoints."""
        assert self.interior_times(t**3 + t, -2, 2) == []

    def test_critical_point_outside_interval_is_excluded(self):
        """Critical points outside (t_start, t_end) are not included."""
        # t**2 has its minimum at 0, to the left of the interval
        assert self.interior_times(t**2, 1, 5) == []

    def test_unsolvable_derivative_falls_back_to_numerics(self):
        """An unsolvable derivative warns and falls back to the numerical search."""
        with pytest.warns(RuntimeWarning, match="falling back to the numerical"):
            ts = TimeSeries.from_function(
                sp.exp(-(t**2)) * sp.sin(5 * t), -2, 2, preserve_extrema=True
            )
        assert len(ts.times) > 2
        assert list(ts.times) == sorted(ts.times)


class TestSampling:
    """Uniform sampling and plain callables."""

    def test_uniform_sampling(self):
        """preserve_extrema=False samples n_points uniformly."""
        ts = TimeSeries.from_function(t**2, 0, 1, n_points=11, preserve_extrema=False)
        assert len(ts.times) == 11
        assert ts.times[0] == pytest.approx(0.0)
        assert ts.times[-1] == pytest.approx(1.0)
        assert ts.values[-1] == pytest.approx(1.0)

    def test_plain_callable_uses_numerical_path(self):
        """A non-sympy callable is handled by the numerical extrema search."""
        ts = TimeSeries.from_function(
            lambda x: np.sin(x), 0, 10, n_points=500, preserve_extrema=True
        )
        assert len(ts.times) > 2
        assert list(ts.times) == sorted(ts.times)


class TestLevelSetTree:
    """Level-set tree construction from a simple excursion."""

    def test_excursion_to_tree(self):
        """A three-peak excursion maps to a valid 6-node tree, one leaf per peak."""
        ts = TimeSeries.from_array([0, 2, 1, 3, 1, 4, 0])
        _minima, maxima = ts.find_local_extrema()
        assert list(maxima) == [1, 3, 5]

        tree = ts.to_level_set_tree()
        assert tree.n_nodes == 6
        assert tree.validate()
        assert tree.get_leaves() == [1, 3, 5]  # one leaf per local maximum
        assert list(tree.parent) == [-1, 2, 0, 4, 2, 4]
        assert tree.horton_strahler_order_tree() == 2
