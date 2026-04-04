Bézier Simplex Splitting
========================

The ``torch_bsf.splitting`` module provides functions to subdivide a fitted
Bézier simplex into two sub-simplices that together cover the entire original
parameter domain.  Splitting is useful whenever a single global Bézier simplex
does not capture all the local features of a manifold — by recursively splitting
the problematic region you can achieve higher accuracy without increasing the
polynomial degree.

.. code-block:: python

   import torch
   import torch_bsf
   from torch_bsf.splitting import split, split_by_criterion, longest_edge_criterion

   # Fit an initial model
   params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
   values = torch.tensor([[0.0], [1.0], [0.0]])
   bs = torch_bsf.fit(
       params=params,
       values=values,
       degree=2,
       max_epochs=200,
       enable_progress_bar=False,
       logger=False,
       enable_checkpointing=False,
   )

   # Split along the longest edge in value space
   bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)

How Splitting Works
-------------------

Given a Bézier simplex :math:`B` of degree :math:`n` over a simplex parameter
domain with vertices :math:`v_0, \ldots, v_{m-1}`, splitting along edge
:math:`(i, j)` at position :math:`s \in (0, 1)` inserts a new vertex

.. math::

   v_\mathrm{new} = (1-s)\,v_i + s\,v_j

and produces two sub-simplices:

* **bs_A** — replaces vertex :math:`j` with :math:`v_\mathrm{new}`.  It covers the
  sub-domain :math:`\{t : t_j / (t_i + t_j) \le s\}`.
* **bs_B** — replaces vertex :math:`i` with :math:`v_\mathrm{new}`.  It covers the
  sub-domain :math:`\{t : t_j / (t_i + t_j) \ge s\}`.

Both sub-simplices have the **same degree and number of parameters** as the
original and together reproduce it exactly: for any point :math:`t` in the
original domain, evaluating the appropriate sub-simplex at its local
coordinates gives the same value as evaluating the original.

The control points of the two sub-simplices are computed using the **de
Casteljau algorithm** applied along the chosen edge direction.  At each step
:math:`r = 1, \ldots, n`:

.. math::

   c^{(r)}_\alpha =
   \begin{cases}
       s \cdot c^{(r-1)}_\alpha
           + (1-s) \cdot c^{(r-1)}_{\alpha + e_i - e_j}
       & \text{if } \alpha_j \ge 1, \\
       c^{(r-1)}_\alpha & \text{otherwise.}
   \end{cases}

Rows with :math:`\alpha_j = r` are saved as control points of **bs_A**.  An
analogous recursion with the roles of :math:`i` and :math:`j` swapped gives
the control points of **bs_B**.

Splitting a Specific Edge
--------------------------

Use :func:`~torch_bsf.splitting.split` when you know which edge to split and
where:

.. code-block:: python

   from torch_bsf.splitting import split

   # Split edge (0, 1) of a Bézier triangle at the midpoint
   bs_A, bs_B = split(bs, i=0, j=1, s=0.5)

   # Evaluate each sub-simplex on a fine grid
   from torch_bsf.sampling import simplex_grid
   t_fine = simplex_grid(n_params=2, degree=20).float()
   pred_A = bs_A(t_fine)
   pred_B = bs_B(t_fine)

The ``s`` parameter controls where the new vertex is placed on the edge:

* ``s = 0.5`` (default) — midpoint split.
* ``s < 0.5`` — the new vertex is closer to :math:`v_i`; **bs_A** covers a
  smaller sub-domain.
* ``s > 0.5`` — the new vertex is closer to :math:`v_j`; **bs_B** covers a
  smaller sub-domain.

Re-parameterising Data Points
------------------------------

After splitting, existing data points need to be mapped to the local
barycentric coordinates of the sub-simplex they belong to.
:func:`~torch_bsf.splitting.reparametrize` performs this conversion:

.. code-block:: python

   from torch_bsf.splitting import reparametrize

   # Original parameter vectors, shape (N, n_params)
   t = params  # e.g. from your training dataset

   # Map to sub-simplex A
   u_A, mask_A = reparametrize(t, i=0, j=1, s=0.5, subsimplex="A")
   # u_A[mask_A] are the local coordinates of the points in sub-simplex A

   # Map to sub-simplex B
   u_B, mask_B = reparametrize(t, i=0, j=1, s=0.5, subsimplex="B")

The returned ``mask`` is a boolean tensor indicating which input points belong
to the requested sub-simplex.  Points where :math:`t_i = t_j = 0` lie on a
shared boundary and appear in both masks.

.. note::

   The local barycentric coordinates ``u`` returned by
   :func:`~torch_bsf.splitting.reparametrize` sum to 1 by construction, so
   they can be passed directly to :meth:`~torch_bsf.BezierSimplex.__call__`.

Choosing a Split Automatically
-------------------------------

Rather than specifying ``(i, j, s)`` by hand, you can pass a
:data:`~torch_bsf.splitting.SplitCriterion` to
:func:`~torch_bsf.splitting.split_by_criterion`.  The criterion inspects the
fitted model (and optionally the data) and returns the best ``(i, j, s)``
triple.

Two built-in criteria are provided.

Longest-Edge Criterion
^^^^^^^^^^^^^^^^^^^^^^

:func:`~torch_bsf.splitting.longest_edge_criterion` selects the edge with the
greatest distance in **value space** and always splits at the midpoint:

.. math::

   \ell_{ij} = \|B(e_i) - B(e_j)\|_2

This criterion requires **only the fitted model** and is very fast.  It works
well when the manifold is elongated along a single direction.

.. code-block:: python

   from torch_bsf.splitting import longest_edge_criterion, split_by_criterion

   bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)

You can also call the criterion directly to inspect which edge it selects:

.. code-block:: python

   i, j, s = longest_edge_criterion(bs)
   print(f"Splitting edge ({i}, {j}) at s={s}")

Maximum-Error Criterion
^^^^^^^^^^^^^^^^^^^^^^^

:func:`~torch_bsf.splitting.max_error_criterion` finds the edge and split
position that minimise the combined mean-squared error over the training data:

.. math::

   E(i, j, s) = \mathrm{MSE}_A + \mathrm{MSE}_B

A grid search over candidate split positions in :math:`(0, 1)` is performed
for every edge.  This criterion is **data-driven** and typically produces a
better split than the longest-edge criterion, at the cost of additional
computation.

.. code-block:: python

   from torch_bsf.splitting import max_error_criterion, split_by_criterion

   # Build the criterion (binds the training data)
   criterion = max_error_criterion(params, values, grid_size=10)

   # Split the model using the data-driven criterion
   bs_A, bs_B = split_by_criterion(bs, criterion)

The ``grid_size`` parameter controls how many candidate ``s`` values are
evaluated per edge.  Larger values give a finer search at the cost of more
forward passes through the model.

Custom Criteria
^^^^^^^^^^^^^^^

A :data:`~torch_bsf.splitting.SplitCriterion` is any callable with the
signature ``(bs: BezierSimplex) -> tuple[int, int, float]``.  You can write
your own criterion and pass it to :func:`~torch_bsf.splitting.split_by_criterion`:

.. code-block:: python

   from torch_bsf import BezierSimplex

   def my_criterion(bs: BezierSimplex) -> tuple[int, int, float]:
       # Always split the first edge at the one-third point
       return 0, 1, 1.0 / 3.0

   bs_A, bs_B = split_by_criterion(bs, my_criterion)

Iterative Refinement
---------------------

A common workflow is to fit an initial model and then refine it by repeatedly
splitting the sub-simplex with the largest approximation error.  The example
below demonstrates one round of such refinement:

.. code-block:: python

   import torch
   import torch_bsf
   from torch_bsf.splitting import (
       max_error_criterion,
       reparametrize,
       split_by_criterion,
   )

   # ── 1. Initial data and fit ────────────────────────────────────────────────
   params = torch.tensor([
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
       [0.5, 0.5, 0.0],
       [0.5, 0.0, 0.5],
       [0.0, 0.5, 0.5],
   ])
   values = torch.stack([p[0:1] * p[1:2] for p in params])  # nonlinear target

   bs = torch_bsf.fit(
       params=params,
       values=values,
       degree=2,
       max_epochs=500,
       enable_progress_bar=False,
       logger=False,
       enable_checkpointing=False,
   )

   # ── 2. Split using the data-driven criterion ───────────────────────────────
   criterion = max_error_criterion(params, values, grid_size=10)
   bs_A, bs_B = split_by_criterion(bs, criterion)
   i, j, s = criterion(bs)  # retrieve the chosen (i, j, s) for reparametrisation

   # ── 3. Reparametrize the data and re-fit each sub-simplex ─────────────────
   u_A, mask_A = reparametrize(params, i, j, s, subsimplex="A")
   u_B, mask_B = reparametrize(params, i, j, s, subsimplex="B")

   if mask_A.sum() > 0:
       bs_A_refined = torch_bsf.fit(
           params=u_A[mask_A],
           values=values[mask_A],
           degree=2,
           max_epochs=500,
           enable_progress_bar=False,
           logger=False,
           enable_checkpointing=False,
       )
   if mask_B.sum() > 0:
       bs_B_refined = torch_bsf.fit(
           params=u_B[mask_B],
           values=values[mask_B],
           degree=2,
           max_epochs=500,
           enable_progress_bar=False,
           logger=False,
           enable_checkpointing=False,
       )

API Reference
-------------

See the following entries in the `API Documentation <../modules.html>`_ for
the full parameter reference:

* :func:`torch_bsf.splitting.split`
* :func:`torch_bsf.splitting.reparametrize`
* :func:`torch_bsf.splitting.longest_edge_criterion`
* :func:`torch_bsf.splitting.max_error_criterion`
* :func:`torch_bsf.splitting.split_by_criterion`
* :data:`torch_bsf.splitting.SplitCriterion`
