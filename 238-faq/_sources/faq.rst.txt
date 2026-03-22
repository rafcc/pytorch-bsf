Frequently asked questions
==========================

There are already many tools for hyperparameter search in ML. Why propose yet another one?
-------------------------------------------------------------------------------------------

The key difference is that PyTorch-BSF **exploits problem structure** rather than treating the objective as a black box.

**Dramatically fewer evaluations.**
Black-box methods such as Bayesian optimization make no assumptions about the objective and must explore the search space from scratch.
Approximating a Pareto front to reasonable accuracy can require hundreds of evaluations with such methods.
Because PyTorch-BSF assumes the problem is *weakly simplicial* — meaning the Pareto set has a simplex structure — it can recover the entire Pareto front from as few as 50 points, often with higher accuracy.

**Dimension-free convergence in the manifold setting.**
When the data lie along a low-dimensional simplex embedded in a high-dimensional ambient space (analogous to manifold learning), the asymptotic risk of Bézier simplex fitting depends only on the **intrinsic dimension** of the simplex, not on the ambient space dimension.
This gives extremely fast statistical convergence in settings where black-box methods would suffer from the curse of dimensionality.

**The weakly simplicial assumption is broadly applicable.**
While the assumption may sound restrictive, it in fact covers a wide class of practical problems — including all unconstrained strongly convex problems such as elastic net regression.
If you are unsure whether your problem qualifies, it can be verified with a statistical test; see `Can I verify whether my problem is weakly simplicial before fitting?`_ below.


Are approximation results always reliable?
------------------------------------------

Not always.
The approximation theorem makes no guarantees about the accuracy of a Bézier simplex of **fixed** degree.
After fitting, you should assess the goodness of fit using domain knowledge appropriate to your problem.


Are there any applications other than multiobjective optimization?
------------------------------------------------------------------

Not yet documented, but quite possibly yes.
Bézier simplices are likely to arise wherever high-dimensional shape representation is needed.
If you discover a new application, please share it — it will be added to the applications section.


How do I choose the degree of the Bézier simplex?
--------------------------------------------------

There is no single rule, but the following guidelines are helpful:

- **Start low.** A degree-2 or degree-3 simplex is a good default. Low-degree models are less prone to overfitting and faster to fit.
- **Check the residuals.** If the fit is visibly poor, increase the degree by one and refit. Repeat until the residuals are acceptable.
- **Mind the sample size.** A Bézier simplex of degree :math:`D` over an :math:`(M-1)`-dimensional simplex has :math:`\binom{D+M-1}{M-1}` control points. You need at least that many training samples for the problem to be well-determined.
- **Avoid over-fitting.** If the model interpolates training points but predicts poorly elsewhere, the degree is too high for the available data.


How many training samples do I need?
-------------------------------------

At minimum, you need as many samples as there are control points.
A Bézier simplex of degree :math:`D` with :math:`M`-dimensional parameters has :math:`\binom{D+M-1}{M-1}` control points.
For example, a degree-3 model with 2-dimensional parameters (a Bézier curve) has 4 control points, while the same degree with 3-dimensional parameters (a Bézier triangle) has 10.

In practice, having two to three times as many samples as control points tends to give stable fits.
If you have fewer samples than control points, consider reducing the degree.


What if my input parameters do not lie on a simplex?
-----------------------------------------------------

The ``fit()`` function requires that each row of the ``params`` tensor sums to 1 (i.e., lies on the standard simplex :math:`\Delta^{M-1}`).
If your raw hyperparameters do not satisfy this constraint, you need to normalize them first.
For two parameters :math:`a` and :math:`b`, a common approach is to set :math:`t_1 = a / (a + b)` and :math:`t_2 = b / (a + b)`.
More generally, divide each parameter vector by its :math:`\ell_1` norm.


Can I train on a GPU?
---------------------

Yes. PyTorch-BSF is built on PyTorch, so GPU acceleration works in the standard way.
Move your tensors to the target device before calling ``fit()``:

.. code-block:: python

   import torch
   import torch_bsf

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ts = ts.to(device)
   xs = xs.to(device)
   bs = torch_bsf.fit(params=ts, values=xs, degree=3)


What should I do if fitting does not converge or the accuracy is poor?
----------------------------------------------------------------------

Try the following steps in order:

1. **Check your data.** Verify that ``params`` rows sum to 1 and that the number of samples is at least as large as the number of control points.
2. **Adjust the degree.** If the fit is underdetermined, reduce the degree. If the Pareto front has high curvature, increase it.
3. **Provide a better initialization.** Use the ``init`` argument of ``fit()`` to supply initial control points derived from domain knowledge or a coarser prior fit.
4. **Increase training iterations.** The underlying L-BFGS optimizer may need more steps for complex surfaces.
5. **Inspect the residuals.** Large residuals concentrated at specific parameter values suggest that more training samples are needed in that region.


When should I use partial training (the ``fix`` argument)?
----------------------------------------------------------

The ``fix`` argument lets you hold specific control points constant while optimizing the rest.
This is useful in several situations:

- **Boundary constraints.** If you know the values at the vertices of the simplex exactly (e.g., single-objective optima), fix those control points and fit only the interior.
- **Incremental refinement.** Fit a coarse model first, then fix the already-well-estimated control points and refine the rest with additional data.
- **Prior knowledge.** If physical or theoretical considerations pin down part of the surface, encoding that as fixed control points prevents the optimizer from moving away from known-good values.

See the :doc:`advanced` section for a code example.


Can I verify whether my problem is weakly simplicial before fitting?
--------------------------------------------------------------------

For a broad and practically important class of problems, no verification is needed: it is known that **all unconstrained strongly convex optimization problems are weakly simplicial** [3].
This covers, for example, elastic net regression and many other regularized empirical risk minimization problems.

For problems outside this class, a data-driven statistical test exists (see [4] in the :doc:`whatis` references).
The test checks whether the topology of the empirical Pareto set is consistent with a simplex structure.
If the test rejects the simplicial hypothesis, a Bézier simplex model may not be appropriate, and you should consider a more general surrogate model.
If the test does not reject, you have statistical evidence supporting the use of PyTorch-BSF.

.. [3] Mizota, Y., Hamada, N., & Ichiki, S. (2021). *All unconstrained strongly convex problems are weakly simplicial.* arXiv:2106.12704. https://arxiv.org/abs/2106.12704


What kinds of shapes can a Bézier simplex represent beyond Pareto fronts?
-------------------------------------------------------------------------

Any continuous map from a simplex to a Euclidean space can be approximated by a Bézier simplex (approximation theorem).
Beyond Pareto fronts, potential applications include:

- **Interpolation of parametric families.** When a model's behavior varies continuously with a set of coefficients that live on a simplex (e.g., mixture weights, regularization strengths), a Bézier simplex can compactly represent the entire family.
- **Shape modeling.** Bézier simplices generalize Bézier triangles used in surface modeling; they can represent smooth curved surfaces of any dimension.
- **Solution manifolds.** Any problem whose solution set forms a continuous simplex-structured manifold — not just multiobjective optimization — is a candidate.

If you find a new application, please share it so it can be documented in the applications section.
