Active Learning
===============

When training costs are high (e.g., in engineering simulations), you want to achieve the best possible Bézier simplex fit with as few samples as possible. **Active Learning** helps you achieve this by suggesting where on the simplex to collect the next sample.

PyTorch-BSF provides several strategies for suggesting these points.

Methods
-------

The ``suggest_next_points()`` function in the ``torch_bsf.active_learning`` module provides two primary strategies.

Query-By-Committee (QBC)
^^^^^^^^^^^^^^^^^^^^^^^^

This method evaluates an ensemble (committee) of models. It suggests points where the different models in the ensemble disagree most (i.e., have the highest prediction variance).

*   **When to use**: When you can afford to train a few models (e.g., via k-fold CV) or want to find areas where the Bézier manifold is "unstable."

.. code-block:: python

   from torch_bsf.active_learning import suggest_next_points

   # Assume an ensemble of trained models (models: List[BezierSimplex])
   suggestions = suggest_next_points(models, n_suggestions=5, method="qbc")

Density-Based Sampling
^^^^^^^^^^^^^^^^^^^^^^

This method suggests points that are furthest away from all existing training points in the parameter space. It is equivalent to a "max-min" distance strategy to ensure even coverage across the entire simplex.

*   **When to use**: When you have only one model or want to avoid clustering samples in one area.

.. code-block:: python

   from torch_bsf.active_learning import suggest_next_points

   # existing_params: (n_samples, n_params)
   suggestions = suggest_next_points(models, n_suggestions=5, method="density", params=existing_params)

A Typical Workflow
------------------

Active learning is usually performed in an iterative loop:

1.  **Initial Sampling**: Start with a small set of initial points (e.g., using grid or random sampling).
2.  **Model Training**: Fit an ensemble of Bézier simplices (using ``KFoldTrainer`` or multiple ``fit()`` calls).
3.  **Suggestion**: Use ``suggest_next_points(method="qbc")`` to find the top 5 areas of highest uncertainty.
4.  **Target Sampling**: Perform your expensive simulation or evaluation at these 5 suggested parameter points.
5.  **Iteration**: Add the new data and go back to step 2 until the desired accuracy is achieved.

Advanced Configuration
----------------------

*   ``n_candidates``: The number of random points on the simplex that are evaluated internally to find the "best" suggestions. Increasing this (e.g., to 10000) will yield more precise suggestions but takes more time.

API Reference
-------------

See the `API Documentation <../modules.html#torch_bsf.active_learning.suggest_next_points>`_ for details.
