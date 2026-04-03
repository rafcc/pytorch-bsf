"""Bézier simplex splitting via the de Casteljau algorithm.

This module provides functions to split a Bézier simplex along a single edge
by inserting a new vertex and applying the de Casteljau algorithm to compute
the control points of the two resulting sub-simplices.

The split point can be chosen explicitly or determined automatically by
optimising a :data:`SplitCriterion`.

Examples
--------
Split a Bézier curve at its midpoint:

>>> import torch
>>> from torch_bsf.bezier_simplex import rand
>>> from torch_bsf.splitting import split
>>> bs = rand(n_params=2, n_values=3, degree=3)
>>> bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
>>> bs_A.n_params == bs.n_params and bs_A.degree == bs.degree
True

Use the longest-edge criterion to choose the split automatically:

>>> from torch_bsf.splitting import longest_edge_criterion, split_by_criterion
>>> bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from torch_bsf.bezier_simplex import BezierSimplex

#: Type of a split criterion: a callable that accepts a
#: :class:`~torch_bsf.bezier_simplex.BezierSimplex` and returns
#: ``(i, j, s)`` — the edge vertex indices and the split parameter.
SplitCriterion = Callable[[BezierSimplex], tuple[int, int, float]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _precompute_shift_rows(
    indices: list[tuple[int, ...]],
    index_to_row: dict[tuple[int, ...], int],
    i: int,
    j: int,
    direction: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the row-index shift table for one de Casteljau direction.

    Parameters
    ----------
    indices
        All control-point indices in canonical order.
    index_to_row
        Mapping from index tuple to matrix row.
    i, j
        Edge vertex indices.
    direction
        ``"ij"`` → look up row of ``alpha + e_i - e_j`` (needs ``alpha_j >= 1``);
        ``"ji"`` → look up row of ``alpha + e_j - e_i`` (needs ``alpha_i >= 1``).
    device
        The device on which the returned tensor is allocated.  Defaults to the
        PyTorch default device (CPU) when ``None``.

    Returns
    -------
    torch.Tensor
        1-D ``long`` tensor of shape ``(n_rows,)``.
        ``shift[r] == -1`` when the shifted index is outside the simplex.
    """
    n = len(indices)
    if direction not in ("ij", "ji"):
        raise ValueError(
            f"`direction` must be 'ij' or 'ji', but got {direction!r}."
        )
    # Build shift values in a plain Python list to avoid per-element kernel
    # launches on CUDA/MPS; create the tensor in one shot at the end.
    shift_list: list[int] = [-1] * n
    for row, alpha in enumerate(indices):
        if direction == "ij":
            if alpha[j] >= 1:
                shifted = list(alpha)
                shifted[i] += 1
                shifted[j] -= 1
                shift_list[row] = index_to_row[tuple(shifted)]
        elif direction == "ji":
            if alpha[i] >= 1:
                shifted = list(alpha)
                shifted[j] += 1
                shifted[i] -= 1
                shift_list[row] = index_to_row[tuple(shifted)]
    return torch.tensor(shift_list, dtype=torch.long, device=device)


def _split_core(
    b: torch.Tensor,
    indices: list[tuple[int, ...]],
    n: int,
    alpha_i: torch.Tensor,
    alpha_j: torch.Tensor,
    shift_ij: torch.Tensor,
    shift_ji: torch.Tensor,
    s: float,
    smoothness_weight: float,
) -> tuple[BezierSimplex, BezierSimplex]:
    """Run the de Casteljau split using pre-computed per-edge tensors.

    Accepts already-computed ``alpha_i/alpha_j`` vectors and shift-row tables
    so that :func:`max_error_criterion` can reuse them across candidate split
    parameters ``s`` for the same edge ``(i, j)`` without redundant work.

    Parameters
    ----------
    b
        Control-point matrix of shape ``(n_rows, n_values)``.
    indices
        All control-point indices in canonical order.
    n
        Degree of the Bézier simplex.
    alpha_i, alpha_j
        1-D long tensors of shape ``(n_rows,)`` holding the ``i``- and
        ``j``-components of each multi-index.
    shift_ij, shift_ji
        Shift-row tables returned by :func:`_precompute_shift_rows` for
        directions ``"ij"`` and ``"ji"`` respectively.
    s
        Split parameter in ``(0, 1)``.
    smoothness_weight
        Passed through to the :class:`~torch_bsf.bezier_simplex.BezierSimplex`
        constructor.
    """
    result_A = torch.empty_like(b)
    result_B = torch.empty_like(b)

    # ---- Sub-simplex A: vertex j → new vertex --------------------------------
    # Recursion:  c^r[alpha] = s * c^{r-1}[alpha]
    #                         + (1-s) * c^{r-1}[alpha + e_i - e_j]  (alpha_j >= 1)
    # Result:     b_A[beta] = c^{beta_j}[beta]
    update_mask_A = shift_ij >= 0  # rows where alpha_j >= 1
    c = b.clone()
    result_A[alpha_j == 0] = c[alpha_j == 0]
    for r in range(1, n + 1):
        c_new = c.clone()
        c_new[update_mask_A] = (
            s * c[update_mask_A] + (1.0 - s) * c[shift_ij[update_mask_A]]
        )
        c = c_new
        result_A[alpha_j == r] = c[alpha_j == r]

    # ---- Sub-simplex B: vertex i → new vertex --------------------------------
    # Recursion:  c^r[alpha] = (1-s) * c^{r-1}[alpha]
    #                         + s * c^{r-1}[alpha + e_j - e_i]       (alpha_i >= 1)
    # Result:     b_B[beta] = c^{beta_i}[beta]
    update_mask_B = shift_ji >= 0  # rows where alpha_i >= 1
    c = b.clone()
    result_B[alpha_i == 0] = c[alpha_i == 0]
    for r in range(1, n + 1):
        c_new = c.clone()
        c_new[update_mask_B] = (
            (1.0 - s) * c[update_mask_B] + s * c[shift_ji[update_mask_B]]
        )
        c = c_new
        result_B[alpha_i == r] = c[alpha_i == r]

    # Build new BezierSimplex instances from result matrices
    cp_data_A: dict[tuple[int, ...], torch.Tensor] = {
        idx: result_A[row] for row, idx in enumerate(indices)
    }
    cp_data_B: dict[tuple[int, ...], torch.Tensor] = {
        idx: result_B[row] for row, idx in enumerate(indices)
    }
    bs_A = BezierSimplex(
        control_points=cp_data_A,
        smoothness_weight=smoothness_weight,
    ).to(device=b.device, dtype=b.dtype)
    bs_B = BezierSimplex(
        control_points=cp_data_B,
        smoothness_weight=smoothness_weight,
    ).to(device=b.device, dtype=b.dtype)
    return bs_A, bs_B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split(
    bs: BezierSimplex,
    i: int,
    j: int,
    s: float = 0.5,
) -> tuple[BezierSimplex, BezierSimplex]:
    r"""Split a Bézier simplex along edge ``(i, j)`` using the de Casteljau algorithm.

    A new vertex is inserted on the edge between vertex ``i`` and vertex ``j``
    of the parameter domain at the relative position ``s``.  The original
    simplex is thereby subdivided into two sub-simplices that together cover
    the entire original domain.

    Parameters
    ----------
    bs
        The Bézier simplex to split.
    i, j
        Indices of the two vertices that define the split edge
        (0-indexed, ``i ≠ j``).
    s
        Split parameter in the open interval ``(0, 1)``.  ``s = 0.5``
        produces a midpoint split.  The new vertex is located at
        :math:`(1-s)\,v_i + s\,v_j` in the parameter domain.

    Returns
    -------
    (bs_A, bs_B) : tuple[BezierSimplex, BezierSimplex]
        * **bs_A** — sub-simplex that replaces vertex :math:`j` with the new
          vertex.  It covers the sub-domain
          :math:`\{t : t_j / (t_i + t_j) \le s\}`.
        * **bs_B** — sub-simplex that replaces vertex :math:`i` with the new
          vertex.  It covers the sub-domain
          :math:`\{t : t_j / (t_i + t_j) \ge s\}`.

    Notes
    -----
    The algorithm runs :math:`n` de Casteljau steps along the chosen edge
    direction, where :math:`n` is the degree of the Bézier simplex.

    At each step :math:`r = 1, \ldots, n` the control-point matrix is updated
    as

    .. math::

        c^{(r)}_\alpha =
        \begin{cases}
            s \cdot c^{(r-1)}_\alpha
                + (1-s) \cdot c^{(r-1)}_{\alpha + e_i - e_j}
            & \text{if } \alpha_j \ge 1 \\
            c^{(r-1)}_\alpha & \text{otherwise,}
        \end{cases}

    and rows with :math:`\alpha_j = r` are saved as control points of
    **bs_A**.  An analogous recursion with the roles of :math:`i` and
    :math:`j` swapped gives **bs_B**.

    The two sub-simplices share the split point — both evaluate to the
    same value at the new vertex.

    Examples
    --------
    Split the identity Bézier curve at the midpoint:

    >>> import torch
    >>> from torch_bsf.bezier_simplex import BezierSimplex
    >>> from torch_bsf.splitting import split
    >>> bs = BezierSimplex({(1, 0): [0.0], (0, 1): [1.0]})
    >>> bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
    >>> float(bs_A.control_points[(0, 1)].item())  # split-point value
    0.5
    >>> float(bs_B.control_points[(1, 0)].item())  # same split point from other side
    0.5
    """
    n_params = bs.n_params
    n = bs.degree
    if n_params < 2:
        raise ValueError(
            f"Splitting requires n_params >= 2, but n_params={n_params}."
        )
    if not (0 <= i < n_params and 0 <= j < n_params and i != j):
        raise ValueError(
            f"Edge indices must satisfy 0 <= i, j < n_params and i != j, "
            f"but i={i}, j={j}, n_params={n_params}."
        )
    if not (0.0 < s < 1.0):
        raise ValueError(f"Split parameter s must be in (0, 1), but s={s}.")

    indices = bs.control_points._indices
    index_to_row = bs.control_points._index_to_row

    b = bs.control_points.matrix

    # alpha_i / alpha_j vectors for mask-based extraction
    alpha_i = torch.tensor(
        [alpha[i] for alpha in indices], dtype=torch.long, device=b.device
    )
    alpha_j = torch.tensor(
        [alpha[j] for alpha in indices], dtype=torch.long, device=b.device
    )

    # Precompute shift-row tables on the same device as the control-point matrix.
    shift_ij = _precompute_shift_rows(indices, index_to_row, i, j, "ij", device=b.device)
    shift_ji = _precompute_shift_rows(indices, index_to_row, i, j, "ji", device=b.device)
    return _split_core(
        b, indices, n, alpha_i, alpha_j, shift_ij, shift_ji, s, bs.smoothness_weight
    )


def reparametrize(
    t: torch.Tensor,
    i: int,
    j: int,
    s: float,
    subsimplex: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Re-parameterise points from the original simplex to a sub-simplex.

    After splitting edge ``(i, j)`` at ``s``, each data point on the original
    simplex belongs to one of the two sub-simplices.  This function converts
    the original barycentric coordinates to the sub-simplex's local
    barycentric coordinates.

    Parameters
    ----------
    t
        Parameter vectors of shape ``(N, n_params)`` on the original simplex
        (each row sums to 1).
    i, j
        Edge vertex indices used in :func:`split`.
    s
        Split parameter.
    subsimplex
        ``"A"`` for the sub-simplex covering
        :math:`\{t : t_j / (t_i + t_j) \le s\}`,
        or ``"B"`` for the complementary region.

    Returns
    -------
    (u, mask) : tuple[torch.Tensor, torch.Tensor]
        * **u** — local barycentric coordinates on the sub-simplex,
          shape ``(N, n_params)``.
        * **mask** — boolean tensor of shape ``(N,)`` indicating which input
          points belong to the requested sub-simplex.  Points where
          :math:`t_i = t_j = 0` belong to both sub-simplices and are included
          in both masks.

    Notes
    -----
    Sub-simplex A transformation:

    .. math::

        u_j = \frac{t_j}{s}, \quad
        u_i = t_i - \frac{1-s}{s}\,t_j, \quad
        u_k = t_k \; (k \ne i, j).

    Sub-simplex B transformation:

    .. math::

        u_i = \frac{t_i}{1-s}, \quad
        u_j = t_j - \frac{s}{1-s}\,t_i, \quad
        u_k = t_k \; (k \ne i, j).
    """
    n_params = t.shape[1]
    if not (0 <= i < n_params and 0 <= j < n_params and i != j):
        raise ValueError(
            f"Edge indices must satisfy 0 <= i, j < n_params and i != j, "
            f"but i={i}, j={j}, n_params={n_params}."
        )
    if not (0.0 < s < 1.0):
        raise ValueError(f"Split parameter s must be in (0, 1), but s={s}.")

    ti = t[:, i]
    tj = t[:, j]
    denom = ti + tj
    safe = denom != 0  # points where t_i + t_j > 0

    u = t.clone()
    if subsimplex == "A":
        # belongs to A when t_j / (t_i + t_j) <= s  (or t_i = t_j = 0)
        mask = (~safe) | (tj <= s * denom)
        u[safe, j] = tj[safe] / s
        u[safe, i] = ti[safe] - (1.0 - s) * tj[safe] / s
    elif subsimplex == "B":
        # belongs to B when t_j / (t_i + t_j) >= s  (or t_i = t_j = 0)
        mask = (~safe) | (tj >= s * denom)
        u[safe, i] = ti[safe] / (1.0 - s)
        u[safe, j] = tj[safe] - s * ti[safe] / (1.0 - s)
    else:
        raise ValueError("subsimplex must be exactly 'A' or 'B'")

    return u, mask


def longest_edge_criterion(
    bs: BezierSimplex,
    s: float = 0.5,
) -> tuple[int, int, float]:
    r"""Select the edge with the greatest value-space length and split it at ``s``.

    The "length" of edge :math:`(i, j)` is the Euclidean distance between
    the Bézier simplex values at the two vertices of the parameter domain:

    .. math::

        \ell_{ij} = \|B(e_i) - B(e_j)\|_2
                  = \|b_{n \cdot e_i} - b_{n \cdot e_j}\|_2

    where :math:`n` is the degree and :math:`e_k` is the :math:`k`-th unit
    vector.

    Parameters
    ----------
    bs
        The Bézier simplex.
    s
        Split parameter included verbatim in the returned tuple
        (defaults to ``0.5``).

    Returns
    -------
    (i, j, s)
        The edge ``(i, j)`` with the greatest value-space length, together
        with ``s``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.bezier_simplex import rand
    >>> from torch_bsf.splitting import longest_edge_criterion, split_by_criterion
    >>> bs = rand(n_params=3, n_values=2, degree=2)
    >>> i, j, s = longest_edge_criterion(bs)
    >>> 0 <= i < j < bs.n_params
    True
    """
    n = bs.degree
    n_params = bs.n_params
    if n_params < 2:
        raise ValueError(
            f"Splitting requires n_params >= 2, but n_params={n_params}."
        )
    b = bs.control_points.matrix.detach()
    index_to_row = {
        idx: row for row, idx in enumerate(bs.control_points.indices())
    }

    # B(e_k) = b[n * e_k]  (the corner control point at vertex k)
    vertex_values = [
        b[index_to_row[tuple(n if m == k else 0 for m in range(n_params))]]
        for k in range(n_params)
    ]
    vertex_matrix = torch.stack(vertex_values, dim=0)

    # Compute all pairwise edge lengths in one shot to avoid repeated
    # scalar extraction and device synchronisation inside nested loops.
    pairwise_dists = torch.cdist(vertex_matrix, vertex_matrix)
    upper_mask = torch.triu(
        torch.ones(
            (n_params, n_params), dtype=torch.bool, device=pairwise_dists.device
        ),
        diagonal=1,
    )
    masked_dists = pairwise_dists.masked_fill(~upper_mask, float("-inf"))
    best_flat = int(torch.argmax(masked_dists).item())
    best_i, best_j = divmod(best_flat, n_params)

    return best_i, best_j, s


def max_error_criterion(
    params: torch.Tensor,
    values: torch.Tensor,
    grid_size: int = 10,
) -> SplitCriterion:
    r"""Build a criterion that minimises the combined approximation error.

    For each candidate edge ``(i, j)`` and split parameter ``s`` drawn from a
    uniform grid over ``(0, 1)``, the combined mean-squared error

    .. math::

        E(i, j, s) = \mathrm{MSE}_A + \mathrm{MSE}_B

    is computed, where :math:`\mathrm{MSE}_A` and :math:`\mathrm{MSE}_B`
    are evaluated on the portions of the data that fall within each
    sub-simplex.  The candidate ``(i, j, s)`` minimising ``E`` is returned.

    Parameters
    ----------
    params
        Parameter vectors of shape ``(N, n_params)``.
    values
        Target value vectors of shape ``(N, n_values)``.
    grid_size
        Number of candidate ``s`` values in the grid search (default ``10``).
        Larger values give a finer search at higher cost.

    Returns
    -------
    SplitCriterion
        A callable ``criterion(bs) -> (i, j, s)``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.bezier_simplex import rand
    >>> from torch_bsf.splitting import max_error_criterion, split_by_criterion
    >>> torch.manual_seed(0)
    <torch._C.Generator object at 0x...>
    >>> params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> values = torch.tensor([[0.0], [0.5], [1.0]])
    >>> bs = rand(n_params=2, n_values=1, degree=1)
    >>> criterion = max_error_criterion(params, values)
    >>> i, j, s = criterion(bs)
    >>> i == 0 and j == 1
    True
    """
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    params_t = torch.as_tensor(params)
    values_t = torch.as_tensor(values)

    def criterion(bs: BezierSimplex) -> tuple[int, int, float]:
        n_params = bs.n_params
        if n_params < 2:
            raise ValueError(
                f"Splitting requires n_params >= 2, but n_params={n_params}."
            )
        if params_t.ndim != 2:
            raise ValueError(
                f"`params` must be a 2D tensor of shape (N, {n_params}), "
                f"but got shape {tuple(params_t.shape)}."
            )
        if values_t.ndim != 2:
            raise ValueError(
                f"`values` must be a 2D tensor of shape (N, {bs.n_values}), "
                f"but got shape {tuple(values_t.shape)}."
            )
        if params_t.shape[0] != values_t.shape[0]:
            raise ValueError(
                f"`params` and `values` must have the same number of samples N, "
                f"but got {params_t.shape[0]} and {values_t.shape[0]}."
            )
        if params_t.shape[1] != n_params:
            raise ValueError(
                f"`params` must have shape (N, {n_params}) for the given "
                f"Bézier simplex, but got shape {tuple(params_t.shape)}."
            )
        if values_t.shape[1] != bs.n_values:
            raise ValueError(
                f"`values` must have shape (N, {bs.n_values}) for the given "
                f"Bézier simplex, but got shape {tuple(values_t.shape)}."
            )
        # Move data to the same device and dtype as the model so that forward
        # passes and mse_loss calls don't raise device/dtype mismatch errors.
        model_device = bs.control_points.matrix.device
        model_dtype = bs.control_points.matrix.dtype
        p = params_t.to(device=model_device, dtype=model_dtype)
        v = values_t.to(device=model_device, dtype=model_dtype)

        best_i, best_j, best_s = 0, 1, 0.5
        best_error = float("inf")
        # linspace(0, 1, grid_size+2)[1:-1] gives `grid_size` evenly-spaced
        # values strictly inside (0, 1), with the midpoint 0.5 included when
        # grid_size is odd (e.g. grid_size=1 → [0.5], grid_size=3 → [0.25, 0.5, 0.75]).
        s_candidates = torch.linspace(0.0, 1.0, grid_size + 2)[1:-1].tolist()

        b = bs.control_points.matrix
        n = bs.degree
        _indices = bs.control_points._indices
        _index_to_row = bs.control_points._index_to_row

        for vi in range(n_params):
            for vj in range(vi + 1, n_params):
                # Precompute alpha and shift tables once per (vi, vj); they
                # are independent of the candidate split position s_cand.
                alpha_vi = torch.tensor(
                    [alpha[vi] for alpha in _indices],
                    dtype=torch.long,
                    device=model_device,
                )
                alpha_vj = torch.tensor(
                    [alpha[vj] for alpha in _indices],
                    dtype=torch.long,
                    device=model_device,
                )
                shift_vivj = _precompute_shift_rows(
                    _indices, _index_to_row, vi, vj, "ij", device=model_device
                )
                shift_vjvi = _precompute_shift_rows(
                    _indices, _index_to_row, vi, vj, "ji", device=model_device
                )
                for s_cand in s_candidates:
                    s_cand = float(s_cand)
                    bs_A, bs_B = _split_core(
                        b, _indices, n,
                        alpha_vi, alpha_vj,
                        shift_vivj, shift_vjvi,
                        s_cand, bs.smoothness_weight,
                    )
                    u_A, mask_A = reparametrize(p, vi, vj, s_cand, "A")
                    u_B, mask_B = reparametrize(p, vi, vj, s_cand, "B")

                    error = 0.0
                    with torch.no_grad():
                        if mask_A.any():
                            pred_A = bs_A(u_A[mask_A])
                            error += float(
                                F.mse_loss(pred_A, v[mask_A]).item()
                            )
                        if mask_B.any():
                            pred_B = bs_B(u_B[mask_B])
                            error += float(
                                F.mse_loss(pred_B, v[mask_B]).item()
                            )

                    if error < best_error:
                        best_error = error
                        best_i, best_j, best_s = vi, vj, s_cand

        return best_i, best_j, best_s

    return criterion


def split_by_criterion(
    bs: BezierSimplex,
    criterion: SplitCriterion,
) -> tuple[BezierSimplex, BezierSimplex]:
    """Split a Bézier simplex using a split criterion.

    Convenience wrapper that calls ``criterion(bs)`` to obtain the edge
    indices and split parameter, then delegates to :func:`split`.

    Parameters
    ----------
    bs
        The Bézier simplex to split.
    criterion
        A :data:`SplitCriterion` callable that returns ``(i, j, s)``.

    Returns
    -------
    (bs_A, bs_B) : tuple[BezierSimplex, BezierSimplex]
        The two sub-Bézier-simplices; see :func:`split` for details.

    Examples
    --------
    >>> from torch_bsf.bezier_simplex import rand
    >>> from torch_bsf.splitting import longest_edge_criterion, split_by_criterion
    >>> bs = rand(n_params=2, n_values=3, degree=2)
    >>> bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
    >>> bs_A.degree == bs.degree
    True
    """
    i, j, s = criterion(bs)
    return split(bs, i, j, s)
