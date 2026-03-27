"""Generate a figure illustrating the leaf-space structure of the elastic-net
hyperparameter space and the corresponding grid sampling on the 2-simplex.

The figure has three panels:

1. The (α, w1) parameter rectangle with the base edge highlighted.
2. The elastic-net grid on the 2-simplex.
3. The quotient space obtained by collapsing the base edge of the simplex to a
   single point (the null model), shown as a leaf/eye-shaped CW complex.

Usage::

    python examples/generate_leaf_space_plot.py

The script writes ``docs/_static/elastic_net_leaf_space.png``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid


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


def to_eye_coords(w1, alpha, r=0.55):
    """Map (w1, alpha) in [0,1]^2 to 2-D leaf/eye coordinates.

    The mapping places vertex A (w1=1, lambda=0) at the top (0, 1) and the
    collapsed base point P* (w1=0, lambda->inf) at the bottom (0, -1).  The
    width of the shape at height corresponding to w1 is proportional to
    sin(pi*(1-w1)), so it tapers to zero at both endpoints producing a
    leaf/eye silhouette.

    Parameters
    ----------
    w1 : float or ndarray
        Data-fidelity weight in [0, 1].
    alpha : float or ndarray
        L1 mixing ratio in [0, 1].
    r : float
        Half-width of the widest cross-section.
    """
    t = 1.0 - w1
    width = r * np.sin(np.pi * t)
    y = np.cos(np.pi * t)
    x = (2.0 * alpha - 1.0) * width
    return x, y


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
# Figure layout: three panels side by side
# ---------------------------------------------------------------------------

fig, (ax_rect, ax_simplex, ax_leaf) = plt.subplots(
    1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.4}
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
for w1 in unique_w1:
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
ax.axhline(y=1.0, color="crimson", linewidth=2.5)
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
# Centre panel – 2-simplex with foliation (leaves) and grid points
# ===========================================================================
ax = ax_simplex

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Elastic-net grid on the 2-simplex\n(after identification)", fontsize=11)

# Triangle boundary
draw_simplex_boundary(ax)

# Highlight base edge (the edge to be identified)
ax.plot([0.5, 1.0], [np.sqrt(3) / 2, 0.0], color="steelblue", lw=2.5, zorder=2)

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

# Label base edge
ax.text(0.8, 0.2, "base edge\n(null model)", fontsize=7.5, ha="center",
        color="steelblue", rotation=-60)

# Colourbar (shared with centre panel)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, aspect=25)
cbar.set_label(r"$w_1 = 1/(1+\lambda)$", fontsize=9)
cbar.ax.tick_params(labelsize=8)


# ===========================================================================
# Right panel – quotient (leaf/eye) space
#
# The 2-simplex has its base edge (connecting (0,1,0) and (0,0,1)) collapsed
# to a single point P*.  The resulting CW complex has:
#   - vertex A  = (1,0,0)  at the top          (lambda = 0)
#   - vertex P* = collapsed base edge           (lambda -> inf, null model)
#   - two 1-cells (edges) from A to P*, shown as curves
#   - one 2-cell (face) enclosed by the two edges
# The overall shape resembles a leaf or eye.
# ===========================================================================
ax = ax_leaf

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Quotient space\n(base edge identified to null model $P^*$)", fontsize=11)

R = 0.55  # half-width of the widest cross-section

# --- Boundary curves ---
# t parametrises the boundary from A (t=0) to P* (t=1)
t_bnd = np.linspace(0.0, 1.0, 300)
x_left_bnd  = -R * np.sin(np.pi * t_bnd)   # alpha=0, left boundary
x_right_bnd =  R * np.sin(np.pi * t_bnd)   # alpha=1, right boundary
y_bnd = np.cos(np.pi * t_bnd)

ax.plot(x_left_bnd,  y_bnd, color="black",     lw=1.5)   # edge AC -> curve to P*
ax.plot(x_right_bnd, y_bnd, color="black",     lw=1.5)   # edge AB -> curve to P*

# --- Leaves: constant-lambda (constant w1) horizontal segments ---
for w1 in unique_w1:
    t = 1.0 - w1
    y_leaf = np.cos(np.pi * t)
    width  = R * np.sin(np.pi * t)
    color  = cmap(1.0 - w1)
    ax.plot([-width, width], [y_leaf, y_leaf], color=color, lw=0.9, alpha=0.7)

# --- Constant-alpha lines: curves from A to P* ---
alpha_lines = np.linspace(0.0, 1.0, N_ALPHAS)
t_line = np.linspace(0.0, 1.0, 200)
for alpha in alpha_lines:
    x_line = (2.0 * alpha - 1.0) * R * np.sin(np.pi * t_line)
    y_line = np.cos(np.pi * t_line)
    ax.plot(x_line, y_line, color="gray", lw=0.5, alpha=0.45)

# --- Grid points in the quotient space ---
n_main = (N_LAMBDAS - 1) * N_ALPHAS
grid_main = grid[:n_main]
w1_main = grid_main[:, 0]
w2_main = grid_main[:, 1]
w3_main = grid_main[:, 2]
w23_sum = w2_main + w3_main
alpha_main = np.where(w23_sum > 1e-10, w2_main / w23_sum, 0.5)
x_eye, y_eye = to_eye_coords(w1_main, alpha_main, r=R)
sc_leaf = ax.scatter(
    x_eye, y_eye,
    c=w1_main, cmap=cmap, vmin=0, vmax=1,
    s=20, zorder=3, edgecolors="none",
)

# --- Vertex markers and labels ---
ax.plot(0, 1, "o", color="crimson",    markersize=7, zorder=5)
ax.plot(0, -1, "o", color="steelblue", markersize=7, zorder=5)

ax.text(0,  1.08, r"$A$: $w_1=1$, $\lambda=0$",
        ha="center", va="bottom", fontsize=8.5, color="crimson")
ax.text(0, -1.10, r"$P^*$: null model ($\lambda\!\to\!\infty$)",
        ha="center", va="top", fontsize=8.5, color="steelblue")

# Edge labels at the widest point
ax.text(-R - 0.04, 0, r"$\alpha=0$" "\n(pure L2)",
        ha="right", va="center", fontsize=8)
ax.text( R + 0.04, 0, r"$\alpha=1$" "\n(pure L1)",
        ha="left",  va="center", fontsize=8)


# ===========================================================================
# Save
# ===========================================================================

fig.savefig(
    "docs/_static/elastic_net_leaf_space.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved docs/_static/elastic_net_leaf_space.png")
