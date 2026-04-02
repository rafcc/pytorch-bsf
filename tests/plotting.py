import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Non-interactive backend for tests

import torch_bsf.bezier_simplex as tbbs
from torch_bsf.plotting import plot_bezier_simplex


@pytest.fixture
def bezier_curve_2d():
    return tbbs.randn(n_params=2, n_values=2, degree=2)


@pytest.fixture
def bezier_curve_3d():
    return tbbs.randn(n_params=2, n_values=3, degree=2)


@pytest.fixture
def bezier_triangle_2d():
    pytest.importorskip("scipy.spatial")
    return tbbs.randn(n_params=3, n_values=2, degree=2)


@pytest.fixture
def bezier_triangle_3d():
    pytest.importorskip("scipy.spatial")
    return tbbs.randn(n_params=3, n_values=3, degree=2)


def test_plot_bezier_curve_2d_returns_axes(bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_curve_3d_returns_axes(bezier_curve_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_3d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_2d_returns_axes(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_3d_returns_axes(bezier_triangle_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_3d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_simplex_high_dim_returns_axes_array():
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert hasattr(result, "shape")
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_3_values():
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=3, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert result.shape == (3, 3)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_no_control_points():
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=5, n_values=2, degree=1)
    result = plot_bezier_simplex(model, num=3, show_control_points=False)
    assert result is not None
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_zero_n_values():
    """n_values == 0 must return an empty (0, 0) ndarray, not (1, 1)."""
    import numpy as np

    model = tbbs.randn(n_params=4, n_values=0, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)


def test_plot_bezier_simplex_pairwise_large_n_values_bounded_figsize():
    """Figure size must be capped at 12 inches even for large n_values."""
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=15, degree=1)
    result = plot_bezier_simplex(model, num=3)
    fig = result[0, 0].figure
    w, h = fig.get_size_inches()
    assert w <= 12.0, f"Figure width {w} exceeds 12-inch cap"
    assert h <= 12.0, f"Figure height {h} exceeds 12-inch cap"
    plt.close(fig)


def test_plot_bezier_simplex_raises_for_low_n_params():
    """plot_bezier_simplex must raise ValueError for n_params < 2."""
    model = tbbs.randn(n_params=1, n_values=2, degree=1)
    with pytest.raises(ValueError, match="n_params"):
        plot_bezier_simplex(model, num=3)


def test_plot_curve_with_existing_axes(bezier_curve_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_curve_2d, num=10, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_triangle_with_existing_axes(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_triangle_2d, num=5, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_curve_without_control_points(bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_triangle_without_control_points(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)
