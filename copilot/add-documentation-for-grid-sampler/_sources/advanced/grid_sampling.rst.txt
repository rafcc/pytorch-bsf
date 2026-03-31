Elastic Net Grid Sampling
=========================

When fitting a Bézier simplex to the elastic-net regularization map you first need to
choose a set of weight vectors on the standard 2-simplex :math:`\Delta^2` at which to
evaluate the model.
The :func:`~torch_bsf.model_selection.elastic_net_grid.elastic_net_grid` function
generates a purpose-built grid that respects the intrinsic geometry of the elastic-net
hyperparameter space.


The Hyperparameter Space
------------------------

The elastic net is conventionally formulated with a regularization strength
:math:`\lambda \ge 0` and an L1/L2 mixing ratio :math:`\alpha \in [0, 1]`:

.. math::

   \min_{\beta} \; f_{\text{data}}(\beta) + \lambda \left(\alpha \, f_{\text{sparse}}(\beta)
   + (1-\alpha) \, f_{\text{smooth}}(\beta)\right).

Here :math:`f_{\text{data}}` is the data-fidelity term, :math:`f_{\text{sparse}}` is the L1
penalty and :math:`f_{\text{smooth}}` is the L2 penalty. This parameterization is a
semi-infinite rectangle :math:`[0,\infty) \times [0,1]` in :math:`(\lambda,\alpha)`.

Equivalently, elastic-net optimization can be written as a convex combination of the
same three objectives:

.. math::

   \min_{\beta} \; w_1 \, f_{\text{data}}(\beta)
               + w_2 \, f_{\text{sparse}}(\beta)
               + w_3 \, f_{\text{smooth}}(\beta),
   \qquad (w_1, w_2, w_3) \in \Delta^2.

The conventional elastic-net parameters :math:`\lambda \ge 0` (overall
regularization strength) and :math:`\alpha \in [0, 1]` (L1 mixing ratio) relate to the
simplex weight vector by:

.. math::

   w_1 = \frac{1}{1+\lambda}, \qquad
   w_2 = \frac{\lambda\,\alpha}{1+\lambda}, \qquad
   w_3 = \frac{\lambda\,(1-\alpha)}{1+\lambda}.

The :math:`(\lambda, \alpha)` parameter space is a semi-infinite rectangle
:math:`[0,\infty) \times [0,1]`.
When :math:`\lambda = 0` the regularization terms vanish and the solution depends only
on the data, *regardless of* :math:`\alpha`.
Therefore the entire edge :math:`\{\lambda = 0\} \times [0, 1]` maps to the single
vertex :math:`(w_1, w_2, w_3) = (1, 0, 0)` of the simplex.
Identifying this edge with a single point transforms the rectangle into a triangle – the 2-simplex :math:`\Delta^2`.

Conversely, as :math:`\lambda \to \infty` the regularization overwhelms the data term
and drives all model coefficients to zero, regardless of :math:`\alpha`.
In the elastic net this limit is called the **null model** (all :math:`\beta_i = 0`).
All weight vectors on the opposite edge of the simplex
:math:`\{(w_1, w_2, w_3) : w_1 = 0\}` — the *base edge* connecting
:math:`(0, 1, 0)` and :math:`(0, 0, 1)` — therefore correspond to the same solution.
Since the Bézier simplex (and the underlying solution map) must assign a single output
to each input weight, all of these base-edge weights are identified with a single
null-model point :math:`P^*` in the solution space. The
:func:`~torch_bsf.model_selection.elastic_net_grid.elastic_net_grid` function still
returns multiple distinct base-edge weights (:math:`w_1 = 0` with varying :math:`w_2`,
:math:`w_3`), but they all evaluate to this same null-model solution. The resulting
quotient space is a **leaf/eye-shaped CW complex**: two 0-cells (:math:`A` and
:math:`P^*`), two 1-cells (the former edges :math:`AB` and :math:`AC`, now connecting
:math:`A` to :math:`P^*` as curves), and one 2-cell (the interior).

This identification gives the interior a **leaf (foliation) structure**: for each fixed
value of :math:`w_1 \in (0, 1]`, the set of corresponding weight vectors
:math:`\{(w_1, w_2, w_3) : w_2 + w_3 = 1 - w_1,\; w_2, w_3 \ge 0\}`
is a line segment (a *leaf*) parametrized by :math:`\alpha`.
As :math:`w_1 \to 0` (i.e. :math:`\lambda \to \infty`), the images of these leaves
under the solution map shrink to the single null-model point :math:`P^*`.

.. figure:: ../_static/elastic_net_leaf_space.png
   :width: 100%

   All points are colored by :math:`(w_1, w_2, w_3) \mapsto (R, G, B)`,
   so the same weight vector has the same color in every panel.
   **Left** – The :math:`(\lambda, \alpha)` hyperparameter space (x: regularization
   strength, y: L1 mixing ratio).
   The red line at :math:`\lambda = 0` is the identified edge; all points on it
   share the color :math:`(1, 0, 0)` = red because :math:`w = (1, 0, 0)` there.
   **Center** – The 2-simplex with vertices :math:`(1,0,0)` at the bottom-left (red),
   :math:`(0,1,0)` at the top (green), and :math:`(0,0,1)` at the
   bottom-right (blue).
   The gradient right edge (green→blue) is the null-model base edge to be identified.
   **Right** – The quotient space rotated 90° counterclockwise: vertex :math:`A` =
   :math:`(1,0,0)` (red) at the left, and the null-model point :math:`P^*` at the
   right shown as a large green dot :math:`(0,1,0)` behind a smaller blue dot
   :math:`(0,0,1)`, reflecting that both endpoints of the base edge are identified
   to :math:`P^*`.


Grid Structure
--------------

A uniform grid in :math:`(\lambda, \alpha)` is sub-optimal because solutions change
rapidly near :math:`\lambda = 0` and slowly for large :math:`\lambda`.
:func:`~torch_bsf.model_selection.elastic_net_grid.elastic_net_grid` therefore uses:

* **Log-scale spacing of the base-edge weight** :math:`w_1` – the
  :func:`~torch_bsf.model_selection.elastic_net_grid.reverse_logspace` routine
  generates ``n_lambdas - 1`` values of :math:`w_1 \in [0, 1)` along the null-model
  base edge, including :math:`w_1 = 0` (which corresponds to :math:`\lambda = \infty`)
  and excluding the data-fidelity vertex at :math:`w_1 = 1`.
  The vertex at :math:`w_1 = 1` (i.e. :math:`\lambda = 0`) is appended separately.
  Since :math:`\lambda = (1 - w_1) / w_1` is finite only for :math:`w_1 > 0`, all
  finite values of :math:`\lambda` arise from :math:`0 < w_1 < 1`.
  This construction produces more samples close to the data-fidelity vertex and
  therefore near :math:`\lambda = 0`.
  The steepness of this clustering is controlled by the ``base`` parameter:
  ``base=1`` gives uniform spacing in :math:`w_1`, while larger values concentrate
  points further towards :math:`w_1 = 1` (i.e. towards smaller :math:`\lambda`).

* **Uniform spacing along** :math:`\alpha` – on each leaf the ``n_alphas`` values of
  :math:`\alpha` are placed uniformly in :math:`[0, 1]`.

The ``n_vertex_copies`` parameter adds extra copies of each simplex vertex.
This is useful when the grid is passed to k-fold cross-validation
(see :doc:`auto_degree`): set ``n_vertex_copies >= k`` for k-fold CV so that,
when the fold-splitting procedure distributes rows approximately evenly, each fold
will contain every vertex at least once.  Using fewer copies than folds can inflate
cross-validation variance.

.. figure:: ../_static/elastic_net_grid_comparison.png
   :width: 100%

   Grid points on the 2-simplex for different values of the ``base`` parameter
   (``n_lambdas=20``, ``n_alphas=10``).  Larger bases push more points towards the
   data-fidelity vertex :math:`(1, 0, 0)` (bottom-left corner in the figure),
   which is appropriate when you expect the optimal model to have small
   :math:`\lambda`.


Usage
-----

As a Python function
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid

   # Generate a grid with 102 lambda levels, 12 alpha levels, and 10 vertex
   # copies (useful for 10-fold cross-validation).
   grid = elastic_net_grid(
       n_lambdas=102,
       n_alphas=12,
       n_vertex_copies=10,
       base=10,
   )
   # grid.shape == (1240, 3)
   # Each row is a weight vector (w1, w2, w3) on the 2-simplex.

   np.savetxt("weights.csv", grid, delimiter=",", fmt="%.17e")

The returned array can be saved to a CSV file and then either (a) loaded back into an
array or tensor and passed as the ``params`` argument to :func:`torch_bsf.fit`, or
(b) passed as a file path to the ``--params`` CLI option, to train a Bézier simplex
over the elastic-net regularization map.

As a Python module (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the module directly to print the grid as a CSV file to *stdout*:

.. code-block:: bash

   python -m torch_bsf.model_selection.elastic_net_grid \
       --n_lambdas=102 \
       --n_alphas=12 \
       --n_vertex_copies=10 \
       --base=10 \
       > weights.csv

All four parameters are optional and fall back to their defaults
(``n_lambdas=102``, ``n_alphas=12``, ``n_vertex_copies=1``, ``base=10``).

Via MLproject
~~~~~~~~~~~~~

The ``elastic_net_grid`` entry point in ``MLproject`` calls the same module and
redirects the output to a CSV file named
``weight_{n_lambdas}_{n_alphas}_{n_vertex_copies}_{base}.csv``:

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf \
       -e elastic_net_grid \
       -P n_lambdas=102 \
       -P n_alphas=12 \
       -P n_vertex_copies=10 \
       -P base=10

After the run completes, the grid is saved in the current working directory as
``weight_102_12_10_10.csv``.  You can then pass it as the ``params`` argument to a
subsequent training run:

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf \
       -P params=weight_102_12_10_10.csv \
       -P values=values.csv \
       -P degree=6

.. seealso::

   * :func:`torch_bsf.model_selection.elastic_net_grid.elastic_net_grid` – API
     reference with parameter descriptions and examples.
   * :func:`torch_bsf.model_selection.elastic_net_grid.reverse_logspace` – helper that
     generates log-spaced samples for the first weight component :math:`w_1`, from
     which the :math:`\lambda = (1 - w_1) / w_1` values are derived.
   * :doc:`auto_degree` – automatic degree selection via k-fold cross-validation.
   * :doc:`../applications/elastic_net` – end-to-end example of elastic-net model
     selection using PyTorch-BSF.
