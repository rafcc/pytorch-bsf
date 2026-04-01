Active Learning
===============

When training costs are high (e.g., in engineering simulations), you want to achieve the best possible Bézier simplex fit with as few samples as possible. **Active Learning** helps you achieve this by suggesting where on the simplex to collect the next sample.

PyTorch-BSF provides two strategies through the ``suggest_next_points()`` function in the ``torch_bsf.active_learning`` module.

Methods
-------

Query-By-Committee (QBC)
^^^^^^^^^^^^^^^^^^^^^^^^

This method trains an *ensemble* (committee) of models and suggests points where the models disagree most—i.e., where the prediction variance across the ensemble is highest.

**When to use QBC:**

*   You already perform k-fold cross-validation as part of your training pipeline (reuse the k fold models at no extra cost).
*   You suspect the manifold has steep or irregular regions that a single model might miss.
*   You want *uncertainty-driven* exploration: focus new samples where the current fit is least confident.

**Practical tips:**

*   A committee of **3–10 models** is usually sufficient. More models give a more reliable uncertainty estimate but increase training time proportionally.
*   With a single model, every point has zero variance, so QBC degenerates to random selection. Use the ``"density"`` method instead in that case.
*   Combine QBC with :doc:`k-fold cross-validation <../advanced/sklearn>` to obtain the ensemble cheaply:

.. code-block:: python

   import torch_bsf
   from torch_bsf.active_learning import suggest_next_points

   # Train an ensemble of 5 models with different random subsets of the data
   models = [
       torch_bsf.fit(params=params_train, values=values_train, degree=3)
       for _ in range(5)
   ]

   # Suggest the 3 most uncertain points
   suggestions = suggest_next_points(models, n_suggestions=3, method="qbc")
   # suggestions: Tensor of shape (3, n_params)

Density-Based Sampling
^^^^^^^^^^^^^^^^^^^^^^

This method suggests points that are as far as possible from **all** existing training points in parameter space—a *max-min distance* (farthest-point) strategy that maximises coverage of the simplex.

**When to use density-based sampling:**

*   You have **only one model**, so QBC's variance estimate is always zero.
*   Your initial samples are clustered in one region and you want to fill the gaps before refining the fit.
*   You are in the **first few iterations** of an active learning loop, where broad coverage matters more than targeting uncertain regions.
*   You want a simple, model-agnostic criterion that does not depend on prediction quality.

**Practical tips:**

*   Pass the full history of sampled parameters as ``params`` so the strategy avoids revisiting already-covered areas.
*   Density sampling is deterministic given the same random seed for candidate generation; QBC results vary with the model ensemble.
*   Switch from density to QBC once you have enough data to train a reliable ensemble (typically after 2–3 initial rounds).

.. code-block:: python

   import torch
   import torch_bsf
   from torch_bsf.active_learning import suggest_next_points

   # Existing training parameters, shape (n_samples, n_params)
   existing_params = torch.tensor([
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
   ])

   model = torch_bsf.fit(params=existing_params, values=existing_values, degree=2)

   # Suggest the 2 points furthest from all existing samples
   suggestions = suggest_next_points(
       [model],
       n_suggestions=2,
       method="density",
       params=existing_params,
   )
   # suggestions: Tensor of shape (2, n_params)

Choosing Sample Sizes
---------------------

Two parameters control how many points are requested and how thoroughly the simplex is searched.

``n_suggestions`` — How Many Points to Add Per Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the **batch size** for each active learning round: how many new simulation runs or experiments you can afford to run before re-training the model.

*   **Small budget (1–3)**: Maximises information per sample but requires frequent re-training. Use when each evaluation is extremely expensive.
*   **Medium budget (5–10)**: A practical default for most engineering workflows. Balances exploration with re-training overhead.
*   **Large budget (20+)**: Useful when evaluations can be parallelised (e.g., batch simulations on a cluster). The gain per sample diminishes as the batch size grows.

A rule of thumb: start with ``n_suggestions`` equal to roughly **10–20% of your current dataset size**, and reduce it as the model converges.

``n_candidates`` — Search Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, ``suggest_next_points()`` draws ``n_candidates`` random points on the simplex and scores each one. Only the top-``n_suggestions`` are returned.

*   **Default (1,000)**: Sufficient for ``n_params ≤ 4`` in most cases.
*   **Increase to 5,000–10,000** when ``n_params`` is large (≥ 5) or when you need more precise placement of suggestions.
*   **Decrease to 200–500** during early prototyping to keep iteration time short; increase to the default 1,000 (or higher) once the workflow is validated.

The tradeoff is purely between **search quality and wall-clock time**: a higher ``n_candidates`` cannot cause incorrect results, only slower ones.

A Complete Active Learning Loop
---------------------------------

The following self-contained example illustrates the full workflow: starting from a handful of corner samples, it iteratively adds points suggested by QBC and re-trains the model.

.. code-block:: python

   import torch
   import torch_bsf
   from torch_bsf.active_learning import suggest_next_points
   from torch_bsf.sampling import simplex_grid


   def expensive_simulation(params: torch.Tensor) -> torch.Tensor:
       """Placeholder for your real evaluation function.

       Parameters
       ----------
       params : Tensor of shape (n, n_params)
           Simplex coordinates to evaluate.

       Returns
       -------
       Tensor of shape (n, n_values)
           Simulated output values.
       """
       # Replace with your actual simulator / experiment
       return torch.sum(params ** 2, dim=1, keepdim=True)


   # ── 1. Initial sampling ────────────────────────────────────────────────────
   # Start with the simplex vertices (degree-1 grid) plus the centroid.
   init_params = simplex_grid(n_params=3, degree=1)           # shape (3, 3)
   centroid = torch.full((1, 3), 1.0 / 3)
   params = torch.cat([init_params, centroid], dim=0)         # shape (4, 3)
   values = expensive_simulation(params)                      # shape (4, 1)

   # ── 2. Active learning loop ────────────────────────────────────────────────
   N_ROUNDS = 5          # number of active learning iterations
   N_SUGGESTIONS = 3     # new points per round
   N_ENSEMBLE = 5        # committee size for QBC
   N_CANDIDATES = 2000   # search resolution

   for round_idx in range(N_ROUNDS):
       # Train an ensemble by fitting N_ENSEMBLE independent models
       ensemble = [
           torch_bsf.fit(params=params, values=values, degree=3, max_epochs=300)
           for _ in range(N_ENSEMBLE)
       ]

       # Suggest the N_SUGGESTIONS most uncertain points using QBC
       next_params = suggest_next_points(
           ensemble,
           n_suggestions=N_SUGGESTIONS,
           n_candidates=N_CANDIDATES,
           method="qbc",
       )

       # Evaluate the suggested points with the expensive function
       next_values = expensive_simulation(next_params)

       # Accumulate the new data
       params = torch.cat([params, next_params], dim=0)
       values = torch.cat([values, next_values], dim=0)

       print(f"Round {round_idx + 1}: dataset size = {params.shape[0]}")

   # ── 3. Final model ─────────────────────────────────────────────────────────
   final_model = torch_bsf.fit(params=params, values=values, degree=3, max_epochs=500)
   print("Final model trained on", params.shape[0], "points.")

**Switching between methods mid-loop**

Start with ``method="density"`` for the first 1–2 rounds to ensure broad simplex coverage, then switch to ``method="qbc"`` once the ensemble has enough data to produce meaningful uncertainty estimates:

.. code-block:: python

   for round_idx in range(N_ROUNDS):
       ensemble = [
           torch_bsf.fit(params=params, values=values, degree=3)
           for _ in range(N_ENSEMBLE)
       ]

       # Use density for initial coverage, QBC once data is sufficient
       if round_idx < 2:
           next_params = suggest_next_points(
               ensemble,
               n_suggestions=N_SUGGESTIONS,
               n_candidates=N_CANDIDATES,
               method="density",
               params=params,
           )
       else:
           next_params = suggest_next_points(
               ensemble,
               n_suggestions=N_SUGGESTIONS,
               n_candidates=N_CANDIDATES,
               method="qbc",
           )

       next_values = expensive_simulation(next_params)
       params = torch.cat([params, next_params], dim=0)
       values = torch.cat([values, next_values], dim=0)

API Reference
-------------

See :func:`torch_bsf.active_learning.suggest_next_points` in the `API Documentation <../modules.html#torch_bsf.active_learning.suggest_next_points>`_ for the full parameter reference.
