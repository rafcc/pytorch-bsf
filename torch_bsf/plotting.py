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
    show_control_points : bool
        Whether to show control points.
    **kwargs
        Additional keyword arguments forwarded to the underlying plot call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    NotImplementedError
        If ``model.n_params`` is not 2 or 3.
    """
    if model.n_params == 2:
        return _plot_bezier_curve(model, num, ax, show_control_points, **kwargs)
    if model.n_params == 3:
        return _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs)
    raise NotImplementedError(f"Plotting for n_params={model.n_params} is not supported.")


def _plot_bezier_curve(model, num, ax, show_control_points, **kwargs):
    """Plots a Bézier curve (n_params == 2).

    For ``model.n_values == 2``, this function produces a 2D plot using
    the first two output components (``xs[:, 0]`` and ``xs[:, 1]``).
    For ``model.n_values >= 3``, this function produces a 3D plot using
    only the first three output components (``xs[:, 0]``, ``xs[:, 1]``,
    and ``xs[:, 2]``); any additional output dimensions are ignored for
    visualization.
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
    matplotlib.axes.Axes
        The axes containing the plot.

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
    matplotlib.axes.Axes
        The axes containing the plot.

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
