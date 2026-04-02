import numpy as np
from torch_bsf.bezier_simplex import BezierSimplex


def plot_bezier_simplex(
    model: BezierSimplex,
    num: int = 100,
    ax=None,
    show_control_points: bool = True,
    **kwargs,
):
    """Plots the Bézier simplex.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points for each edge.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
        Ignored when ``model.n_params >= 4`` (a new figure is always created
        for pairwise plots).
    show_control_points : bool
        Whether to show control points.
    **kwargs
        Additional keyword arguments forwarded to the plot call.
        For ``model.n_params == 2``, forwarded to ``ax.plot`` (curve).
        For ``model.n_params == 3`` and ``model.n_values >= 3``, forwarded
        to ``ax.plot_trisurf`` (3D surface).
        For ``model.n_params == 3`` and ``model.n_values == 2``, ignored.
        For ``model.n_params >= 4``, forwarded to ``ax.scatter`` (pairwise).

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D or numpy.ndarray
        The axes containing the plot.  For ``model.n_params <= 3`` a single
        ``Axes`` (or ``Axes3D``) is returned.  For ``model.n_params >= 4`` a
        2-D ``numpy.ndarray`` of ``Axes`` with shape
        ``(n_values, n_values)`` is returned (pairwise scatter plot).

    Raises
    ------
    ImportError
        If matplotlib is not installed. This dependency is required for all
        plotting backends used by this function.
    ImportError
        If SciPy is not installed and ``model.n_params == 3``. SciPy is
        required for the triangulation-based plotting used in the
        Bézier triangle case.
    """
    if model.n_params == 2:
        return _plot_bezier_curve(model, num, ax, show_control_points, **kwargs)
    if model.n_params == 3:
        return _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs)
    if model.n_params >= 4:
        return _plot_bezier_simplex_pairwise(model, num, show_control_points, **kwargs)
    raise ValueError(
        f"plot_bezier_simplex only supports models with n_params >= 2; got n_params = {model.n_params}"
    )


def _plot_bezier_curve(model, num, ax, show_control_points, **kwargs):
    """Plots a Bézier curve (n_params == 2).

    For ``model.n_values == 2``, this function produces a 2D plot using
    the first two output components (``xs[:, 0]`` and ``xs[:, 1]``).
    For ``model.n_values >= 3``, this function produces a 3D plot using
    only the first three output components (``xs[:, 0]``, ``xs[:, 1]``,
    and ``xs[:, 2]``); any additional output dimensions are ignored for
    visualization.
    For ``model.n_values < 2``, no data is plotted and the axes are
    returned empty.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along the curve.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
    show_control_points : bool
        Whether to overlay the control points.
    **kwargs
        Additional keyword arguments forwarded to the plot call.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot. Returns a 2D ``Axes`` when
        ``model.n_values == 2``, or a 3D ``Axes3D`` when
        ``model.n_values >= 3``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    ts, xs = model.meshgrid(num=num)
    xs = xs.detach().cpu().numpy()

    if ax is None:
        fig = plt.figure()
        if model.n_values >= 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if model.n_values == 2:
        ax.plot(xs[:, 0], xs[:, 1], **kwargs)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], c="r", marker="o", label="Control Points")
            # Connect control points with a dashed line
            ax.plot(cp[:, 0], cp[:, 1], "r--", alpha=0.3)
    elif model.n_values >= 3:
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], **kwargs)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="r", marker="o", label="Control Points")
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], "r--", alpha=0.3)

    return ax


def _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs):
    """Plots a Bézier triangle using triangulation (n_params == 3).

    Depending on ``model.n_values``, this produces either:

    * a 2D triangulated plot plus scattered points when ``model.n_values == 2``, or
    * a 3D triangulated surface plot when ``model.n_values >= 3``.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along each edge of the triangle.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
    show_control_points : bool
        Whether to overlay the control points.
    **kwargs
        Additional keyword arguments forwarded to the 3D surface plot
        (``ax.plot_trisurf``) when ``model.n_values >= 3``.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot. Returns a 2D ``Axes`` when
        ``model.n_values == 2``, or a 3D ``Axes3D`` when
        ``model.n_values >= 3``.

    Raises
    ------
    ImportError
        If matplotlib or scipy is not installed.
    """
    # This requires a bit more complex triangulation for plotting a surface
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise ImportError(
            "scipy is required for triangle plotting. "
            "Install it with: pip install scipy"
        ) from e

    ts, xs = model.meshgrid(num=num)
    xs = xs.detach().cpu().numpy()

    # Project 3D simplex parameters to 2D for triangulation
    # (Using the same projection as in examples)
    t_np = ts.detach().cpu().numpy()
    px = t_np[:, 1] + 0.5 * t_np[:, 2]
    py = np.sqrt(3) / 2 * t_np[:, 2]

    tri = Delaunay(np.stack([px, py], axis=1))

    if ax is None:
        fig = plt.figure()
        if model.n_values >= 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if model.n_values == 2:
        # Use the triangulation connectivity in value space for a consistent 2D plot
        ax.triplot(xs[:, 0], xs[:, 1], tri.simplices, alpha=0.3)
        # We can't easily plot a manifold surface in 2D values unless we pick 2 dimensions
        ax.scatter(xs[:, 0], xs[:, 1], alpha=0.1, s=1)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], c="r", marker="o")
    elif model.n_values >= 3:
        ax.plot_trisurf(
            xs[:, 0], xs[:, 1], xs[:, 2], triangles=tri.simplices, alpha=0.5, **kwargs
        )
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="r", marker="o")

    return ax


def _plot_bezier_simplex_pairwise(model, num, show_control_points, **kwargs):
    """Plots a high-dimensional Bézier simplex as a pairwise scatter plot.

    For ``model.n_params >= 4``, this function generates sample points from
    the simplex and creates a pairwise scatter plot (pair plot) of the output
    values.  Diagonal panels show histograms of individual output dimensions;
    off-diagonal panels show scatter plots of pairs of output dimensions.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along each edge of the simplex.
    show_control_points : bool
        Whether to overlay control points on the off-diagonal scatter panels.
    **kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        A 2-D array of axes with shape ``(n_values, n_values)``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    _ts, xs = model.meshgrid(num=num)
    xs = xs.detach().cpu().numpy()

    n_v = model.n_values
    if n_v == 0:
        fig, single_ax = plt.subplots(1, 1)
        return np.array([[single_ax]])

    panel_size = max(1, min(3, 12 // max(n_v, 1)))
    fig, axes = plt.subplots(
        n_v, n_v, squeeze=False, figsize=(panel_size * n_v, panel_size * n_v)
    )
    cp = (
        model.control_points.matrix.detach().cpu().numpy()
        if show_control_points
        else None
    )

    # Allow caller to override scatter defaults via kwargs
    scatter_s = kwargs.pop("s", 1)
    scatter_alpha = kwargs.pop("alpha", 0.3)
    # Compute a reasonable bin count (Sturges' rule, minimum 10)
    n_samples = len(xs)
    bins = max(10, int(np.ceil(np.log2(n_samples))) + 1) if n_samples > 1 else 10

    for i in range(n_v):
        for j in range(n_v):
            a = axes[i, j]
            if i == j:
                a.hist(xs[:, i], bins=bins)
                if cp is not None:
                    for val in cp[:, i]:
                        a.axvline(val, color="r", alpha=0.5, linewidth=1)
            else:
                a.scatter(xs[:, j], xs[:, i], s=scatter_s, alpha=scatter_alpha, **kwargs)
                if cp is not None:
                    a.scatter(cp[:, j], cp[:, i], c="r", s=20, marker="o", zorder=5)

    return axes
