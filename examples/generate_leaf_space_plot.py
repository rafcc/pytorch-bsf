"""Generate a figure illustrating the leaf-space structure of the elastic-net
hyperparameter space and the corresponding grid sampling on the 2-simplex.

Usage::

    python examples/generate_leaf_space_plot.py

The script writes ``docs/_static/elastic_net_leaf_space.png``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid, reverse_logspace


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_to_2d(points):
    """Project 3-D simplex points to 2-D Cartesian coordinates.

    Vertex mapping::

        (1, 0, 0)  ->  (0,         0        )  # data-only vertex (lambda=0)
        (0, 1, 0)  ->  (1,         0        )  # pure-L1 vertex
        (0, 0, 1)  ->  (0.5, sqrt3/2        )  # pure-L2 vertex
    """
    w1, w2, w3 = points[:, 0], points[:, 1], points[:, 2]
    px = w2 + 0.5 * w3
    py = np.sqrt(3) / 2 * w3
    return px, py


def draw_simplex_boundary(ax, **kwargs):
    """Draw the triangle boundary on *ax*."""
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]],
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        **kwargs,
    )
    ax.add_patch(triangle)


# ---------------------------------------------------------------------------
# Build a coarse grid for illustration (n_lambdas=12, n_alphas=8)
# ---------------------------------------------------------------------------

N_LAMBDAS = 12
N_ALPHAS = 8
BASE = 10

grid = elastic_net_grid(n_lambdas=N_LAMBDAS, n_alphas=N_ALPHAS, base=BASE)
px, py = project_to_2d(grid)

# w1 values for each grid point (needed to colour the leaves)
w1_values = grid[:, 0]

# Unique lambda "levels" present in the grid (excluding the final vertex copy)
unique_w1 = np.unique(grid[: (N_LAMBDAS - 1) * N_ALPHAS, 0])


# ---------------------------------------------------------------------------
# Figure layout: two panels side by side
# ---------------------------------------------------------------------------

fig, (ax_rect, ax_simplex) = plt.subplots(
    1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.35}
)

cmap = plt.cm.viridis_r

# ===========================================================================
# Left panel – (α, w1) rectangle with the base-edge identification
# ===========================================================================
ax = ax_rect

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.10)
ax.set_xlabel(r"$\alpha$  (L1 mixing ratio)", fontsize=11)
ax.set_ylabel(r"$w_1 = \dfrac{1}{1+\lambda}$  (data weight)", fontsize=11)
ax.set_title("Hyperparameter rectangle\n(before identification)", fontsize=11)

# Background rectangle
rect_bg = plt.Polygon(
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    closed=True,
    facecolor="#f5f5ff",
    edgecolor="gray",
    linewidth=0.8,
    linestyle="--",
)
ax.add_patch(rect_bg)

# Draw "leaves" – horizontal lines at each sampled lambda level
for i, w1 in enumerate(unique_w1):
    color = cmap(1.0 - w1)
    ax.axhline(y=w1, color=color, linewidth=0.9, alpha=0.65)

# Draw grid points
alphas_per_row = np.linspace(0.0, 1.0, N_ALPHAS, endpoint=True)
for w1 in unique_w1:
    color = cmap(1.0 - w1)
    ax.scatter(
        alphas_per_row,
        np.full(N_ALPHAS, w1),
        c=[color] * N_ALPHAS,
        s=18,
        zorder=3,
        edgecolors="none",
    )

# Highlight the "identified" base edge (w1 = 1, i.e. lambda = 0)
ax.axhline(y=1.0, color="crimson", linewidth=2.5, label=r"$\lambda = 0$ edge (identified)")
# Tick at vertex copy
ax.scatter([0.0, 1.0], [1.0, 1.0], c="crimson", s=60, zorder=5)

# Bracket / annotation
ax.annotate(
    "identified\nto one point",
    xy=(0.5, 1.0),
    xytext=(0.5, 1.07),
    fontsize=8.5,
    ha="center",
    color="crimson",
    arrowprops=dict(arrowstyle="-[", color="crimson", lw=1.5,
                    connectionstyle="arc3,rad=0"),
)

# Vertex labels
ax.text(-0.04, 0.0, r"$w_1=0$" "\n" r"($\lambda \to \infty$)", fontsize=8, va="center", ha="right", color="gray")
ax.text(-0.04, 1.0, r"$w_1=1$" "\n" r"($\lambda = 0$)", fontsize=8, va="center", ha="right", color="crimson")
ax.text(0.0, -0.04, r"$\alpha=0$" "\n(pure L2)", fontsize=8, va="top", ha="center")
ax.text(1.0, -0.04, r"$\alpha=1$" "\n(pure L1)", fontsize=8, va="top", ha="center")

ax.tick_params(axis="both", which="both", length=3)


# ===========================================================================
# Right panel – 2-simplex with foliation (leaves) and grid points
# ===========================================================================
ax = ax_simplex

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Elastic-net grid on the 2-simplex\n(after identification)", fontsize=11)

# Triangle boundary
draw_simplex_boundary(ax)

# Draw leaves on the simplex
for w1 in unique_w1:
    # A "leaf" at fixed w1 is the segment w2 + w3 = 1 - w1, w2,w3 >= 0
    # Left point: (w1, 0, 1-w1)  =>  project_to_2d -> (0.5*(1-w1), sqrt3/2*(1-w1))
    # Right point: (w1, 1-w1, 0) =>  project_to_2d -> (1-w1, 0)
    x_left = 0.5 * (1 - w1)
    y_left = np.sqrt(3) / 2 * (1 - w1)
    x_right = 1 - w1
    y_right = 0.0
    color = cmap(1.0 - w1)
    ax.plot([x_left, x_right], [y_left, y_right], color=color, lw=0.9, alpha=0.65)

# Scatter grid points (colour by w1)
sc = ax.scatter(
    px, py, c=w1_values, cmap=cmap, vmin=0, vmax=1,
    s=20, zorder=3, edgecolors="none",
)

# Vertex labels
offset = 0.04
ax.text(0 - offset, 0 - offset, r"$w_1=1$" "\n" r"$(\lambda=0)$",
        ha="center", va="top", fontsize=8.5, color="crimson")
ax.text(1 + offset, 0 - offset, r"$w_2=1$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!1)$",
        ha="center", va="top", fontsize=8.5)
ax.text(0.5, np.sqrt(3) / 2 + offset, r"$w_3=1$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!0)$",
        ha="center", va="bottom", fontsize=8.5)

# Colourbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, aspect=25)
cbar.set_label(r"$w_1 = 1/(1+\lambda)$", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ===========================================================================
# Save
# ===========================================================================

fig.savefig(
    "docs/_static/elastic_net_leaf_space.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved docs/_static/elastic_net_leaf_space.png")
