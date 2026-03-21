Elastic net model selection
===========================

A canonical and highly practical application of this theory is hyperparameter optimization for the **Elastic Net**. In machine learning and statistics, selecting the correct regularization parameters is critical to balancing data fidelity, model sparsity, and numerical stability. 

Traditional approaches rely on exhaustive grid searches or cross-validation over a discrete set of hyperparameter combinations :math:`(\alpha, \lambda)`. This process is computationally expensive, as it requires training a separate model for every grid point. By formulating the hyperparameter tuning as a multi-objective optimization problem over the Pareto set :cite:p:`bonnel2019post`, PyTorch-BSF allows for a continuous exploration of the model space, often reducing the computational cost by orders of magnitude (e.g., achieving speeds up to 2,000 times faster than exhaustive search for a 3-objective problem).

Problem Formulation
-------------------

The underlying multi-objective problem involves simultaneously minimizing three objectives over the model weights :math:`\beta \in \mathbb{R}^N`:

.. math:: 
   
   f_{data}(\beta) &= \frac{1}{2}\|y - X\beta\|^2 \\
   f_{sparse}(\beta) &= \|\beta\|_1 \\
   f_{smooth}(\beta) &= \frac{1}{2}\|\beta\|_2^2

These values are combined using the *weighted-sum scalarization* :math:`x^*: \Delta^{M-1}\to\mathbb R^N`:

.. math:: x^*(w)=\arg\min_\beta \sum_{m=1}^3 w_m f_m(\beta).

Because of the :math:`L_2` penalty term (:math:`f_{smooth}`), the overall objective is strongly convex for any :math:`w` where :math:`w_3 > 0`. Specifically, the Hessian of the smooth component is lower bounded by :math:`w_3 I \succ 0`. This ensures that the solution is unique for each weight vector :math:`w`, guaranteeing that the problem is weakly simplicial :cite:p:`mizota2021unconstrained`.

Hyperparameter Mapping
----------------------

The standard Elastic Net objective is typically parameterized as:

.. math:: \min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda \alpha \|\beta\|_1 + \frac{\lambda(1-\alpha)}{2} \|\beta\|_2^2

where :math:`\lambda` controls the overall regularization strength and :math:`\alpha \in [0, 1]` controls the balance between :math:`L_1` and :math:`L_2` penalties. These correspond directly to the weights :math:`w = (w_1, w_2, w_3)` in our multi-objective formulation:

.. math:: 
   w_1 &= \frac{1}{1 + \lambda} \\
   w_2 &= \frac{\lambda \alpha}{1 + \lambda} \\
   w_3 &= \frac{\lambda (1-\alpha)}{1 + \lambda}

By training the model on a sparse subset of weight vectors :math:`w` and fitting a Bézier simplex, we obtain a continuous **solution map** :math:`(x^*, f \circ x^*): \Delta^{M-1} \to G^*(f)` that maps any weight :math:`w` to the optimal weights :math:`\beta` and the corresponding objective values.

Solution Map and Continuous Exploration
---------------------------------------

Fitting a Bézier simplex to the resulting trained models yields a continuous performance surface. This analytic surrogate allows practitioners to:

1. **Instantly explore** the full continuous spectrum of hyperparameters without retraining.
2. **Visualize the Pareto Front** to understand the trade-offs between accuracy, sparsity, and stability.
3. **Analytically locate** the statistically optimal model (e.g., via cross-validation or information criteria) over a continuous domain.

Empirical Validation
--------------------

The effectiveness of PyTorch-BSF for Elastic Net has been validated on various benchmark datasets from the UCI Machine Learning Repository, including:

* **Blog Feedback**: Regression task with high-dimensional features.
* **QSAR Fish Toxicity**: Predicting aquatic toxicity.
* **Slice Localization**: Estimating the relative location of CT slices.
* **Residential Building**: Predicting sale prices based on building attributes.

Experiments show that even with a limited number of training points (e.g., 51 points for a degree-3 simplex), the Bézier simplex accurately approximates the entire solution map, maintaining low Mean Squared Error (MSE) across the continuous hyperparameter space.
