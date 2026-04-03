import pytest

import torch_bsf.bezier_simplex as tbbs
from torch_bsf.plotting import plot_bezier_simplex


@pytest.fixture
def require_matplotlib():
    """Skip the test if matplotlib is not installed and configure the Agg backend."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    return mpl


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


def test_plot_bezier_curve_2d_returns_axes(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_curve_3d_returns_axes(require_matplotlib, bezier_curve_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_3d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_2d_returns_axes(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_3d_returns_axes(require_matplotlib, bezier_triangle_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_3d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_simplex_high_dim_returns_axes_array(require_matplotlib):
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert hasattr(result, "shape")
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_3_values(require_matplotlib):
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=3, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert result.shape == (3, 3)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_no_control_points(require_matplotlib):
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=5, n_values=2, degree=1)
    result = plot_bezier_simplex(model, num=3, show_control_points=False)
    assert result is not None
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_zero_n_values(require_matplotlib):
    """n_values == 0 must return an empty (0, 0) ndarray, not (1, 1)."""
    import numpy as np

    model = tbbs.randn(n_params=4, n_values=0, degree=1)
    result = plot_bezier_simplex(model, num=3)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)


def test_plot_bezier_simplex_pairwise_large_n_values_bounded_figsize(require_matplotlib):
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


def test_plot_bezier_simplex_pairwise_sampling_cap(require_matplotlib):
    """When meshgrid would exceed _MAX_PAIRWISE_POINTS, random sampling is used
    and the result still has the expected (n_values, n_values) shape."""
    import math

    import matplotlib.pyplot as plt
    import torch_bsf.plotting as plotting_module

    # n_params=4, num=30 => binom(33, 3) = 5456, which exceeds _MAX_PAIRWISE_POINTS
    n_params, num = 4, 30
    model = tbbs.randn(n_params=n_params, n_values=2, degree=1)
    assert (
        math.comb(num + n_params - 1, n_params - 1) > plotting_module._MAX_PAIRWISE_POINTS
    ), "precondition: meshgrid size must exceed the cap so random sampling is triggered"
    result = plot_bezier_simplex(model, num=num)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_default(require_matplotlib):
    """Control points exceeding _MAX_CONTROL_POINTS are subsampled by default."""
    import matplotlib.pyplot as plt
    import torch_bsf.plotting as plotting_module

    # degree=20, n_params=4 => comb(23,3) = 1771 control points > _MAX_CONTROL_POINTS (500)
    degree = 20
    model = tbbs.randn(n_params=4, n_values=2, degree=degree)
    import math

    n_cp = math.comb(model.n_params + degree - 1, degree)
    assert (
        n_cp > plotting_module._MAX_CONTROL_POINTS
    ), f"precondition: model must have more than {plotting_module._MAX_CONTROL_POINTS} control points; got {n_cp}"
    result = plot_bezier_simplex(model, num=3)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_custom(require_matplotlib):
    """Custom max_control_points value is respected."""
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=2, degree=5)
    # Set a very small cap so subsampling is always triggered
    result = plot_bezier_simplex(model, num=3, max_control_points=2)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_invalid():
    """plot_bezier_simplex must raise ValueError for negative max_control_points."""
    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    with pytest.raises(ValueError, match="max_control_points"):
        plot_bezier_simplex(model, num=3, max_control_points=-1)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_custom(require_matplotlib):
    """Custom max_pairwise_points value forces random sampling and returns correct shape."""
    import matplotlib.pyplot as plt

    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    # With max_pairwise_points=10, even a small meshgrid will exceed the cap
    result = plot_bezier_simplex(model, num=100, max_pairwise_points=10)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_invalid():
    """plot_bezier_simplex must raise ValueError for negative max_pairwise_points."""
    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    with pytest.raises(ValueError, match="max_pairwise_points"):
        plot_bezier_simplex(model, num=3, max_pairwise_points=-1)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_default(require_matplotlib):
    """Default max_pairwise_points matches _MAX_PAIRWISE_POINTS and returns correct shape."""
    import matplotlib.pyplot as plt
    import torch_bsf.plotting as plotting_module

    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    result = plot_bezier_simplex(model, num=3, max_pairwise_points=plotting_module._MAX_PAIRWISE_POINTS)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_curve_with_existing_axes(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_curve_2d, num=10, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_triangle_with_existing_axes(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_triangle_2d, num=5, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_curve_without_control_points(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_triangle_without_control_points(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# ImportError handling tests (matplotlib / scipy not available)
# ---------------------------------------------------------------------------


def test_plot_bezier_curve_no_matplotlib(monkeypatch):
    """plot_bezier_simplex should raise ImportError when matplotlib is unavailable (curve)."""
    import sys
    from torch_bsf.plotting import _plot_bezier_curve

    model = tbbs.randn(n_params=2, n_values=2, degree=1)
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises((ImportError, TypeError)):
            _plot_bezier_curve(model, 10, None, True)


def test_plot_bezier_triangle_no_matplotlib(monkeypatch):
    """plot_bezier_simplex should raise ImportError when matplotlib is unavailable (triangle)."""
    import sys
    from torch_bsf.plotting import _plot_bezier_triangle

    model = tbbs.randn(n_params=3, n_values=3, degree=1)
    with monkeypatch.context() as m:
        m.delitem(sys.modules, "matplotlib", raising=False)
        m.delitem(sys.modules, "matplotlib.pyplot", raising=False)
        with pytest.raises(ImportError):
            _plot_bezier_triangle(model, 5, None, True)


def test_plot_bezier_triangle_no_scipy(monkeypatch):
    """_plot_bezier_triangle should raise ImportError when scipy is unavailable."""
    import sys
    from torch_bsf.plotting import _plot_bezier_triangle

    model = tbbs.randn(n_params=3, n_values=3, degree=1)
    with monkeypatch.context() as m:
        m.delitem(sys.modules, "scipy", raising=False)
        m.delitem(sys.modules, "scipy.spatial", raising=False)
        with pytest.raises(ImportError, match="scipy"):
            _plot_bezier_triangle(model, 5, None, True)


def test_plot_pairwise_no_matplotlib(monkeypatch):
    """_plot_bezier_simplex_pairwise should raise ImportError when matplotlib is unavailable."""
    import sys
    from torch_bsf.plotting import _plot_bezier_simplex_pairwise

    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises((ImportError, TypeError)):
            _plot_bezier_simplex_pairwise(model, 5, True, 500, 2000)
