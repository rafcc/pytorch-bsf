import numpy as np
from torch_bsf.bezier_simplex import BezierSimplex


def plot_bezier_simplex(
    model: BezierSimplex,
    num: int = 100,
    ax=None,
    show_control_points: bool = True,
    **kwargs,
):
    """Plots the Bezier simplex.

    Parameters
    ----------
    model
        The Bezier simplex model.
    num
        The number of grid points for each edge.
    ax
        The matplotlib axes to plot on.
    show_control_points
        Whether to show control points.
    kwargs
        Additional arguments for the plot.
    """
    if model.n_params == 2:
        return _plot_bezier_curve(model, num, ax, show_control_points, **kwargs)
    if model.n_params == 3:
        return _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs)
    raise NotImplementedError(f"Plotting for n_params={model.n_params} is not supported.")


def _plot_bezier_curve(model, num, ax, show_control_points, **kwargs):
    import matplotlib.pyplot as plt

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
    # This requires a bit more complex triangulation for plotting a surface
    from scipy.spatial import Delaunay

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
        ax.triplot(px, py, tri.simplices, alpha=0.3)
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
