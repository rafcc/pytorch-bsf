Elastic net model selection
===========================

A canonical and highly practical application of this theory is hyperparameter optimization for the **Elastic Net**. In machine learning and statistics, selecting the correct regularization parameters is critical to balancing data fidelity, model sparsity, and numerical stability. 

Traditional approaches rely on exhaustive grid searches or cross-validation over a discrete set of hyperparameter combinations :math:`(\alpha, \lambda)`. This process is computationally expensive. By formulating the hyperparameter tuning as a multi-objective optimization problem over a **regularization map** :cite:p:`bonnel2019post`, PyTorch-BSF allows for a continuous exploration of the model space, often reducing the computational cost by orders of magnitude (e.g., achieving speeds up to 2,000 times faster than exhaustive search for a 3-objective problem).

Unified Framework for Sparse Modeling
--------------------------------------

The Elastic Net example is part of a broader framework for **generalized sparse modeling**. Any optimization problem expressed as a convex combination of several strongly convex functions can be analyzed using this approach:

.. math:: 
   
   \min_{\beta \in \mathbb{R}^N} f_w(\beta) = \sum_{m=1}^M w_m f_m(\beta), \quad w \in \Delta^{M-1}

where :math:`f_m` represent different aspects of the model, such as data fidelity (loss functions) or structural priors (regularization terms). By ensuring each :math:`f_m` is strongly convex (e.g., by adding a small :math:`L_2` penalty :math:`\frac{\epsilon}{2} \|\beta\|_2^2`), the resulting solution map :math:`x^*(w)` is guaranteed to be weakly simplicial :cite:p:`mizota2021unconstrained`, allowing for high-quality approximation via Bézier simplices.

This unified framework extends far beyond the standard Elastic Net:

* **Generalized Linear Models (GLMs)**: The loss term can be any negative log-likelihood with a canonical link function (e.g., Logistic, Poisson, or Gamma regression).
* **Structural Regularization**: Practitioners can incorporate structural priors like **Group Lasso** :cite:p:`yuan2006model`, **Fused Lasso** :cite:p:`tibshirani2005sparsity`, or **Smoothed Lasso** to encode spatial or group relationships between features.
* **Transfer Learning and Covariate Shift**: By using importance-weighted empirical risk as the loss function, the framework can handle scenarios where the training and test data distributions differ (covariate shift).
* **Robust Estimation**: Replacing the standard squared error with robust loss functions like the **Huber loss** allows for model selection that is resilient to outliers.

Elastic Net Formulation
-----------------------

For the standard Elastic Net, the underlying multi-objective problem involves simultaneously minimizing three objectives over the model weights :math:`\beta \in \mathbb{R}^N`:

.. math:: 
   
   f_{data}(\beta) &= \frac{1}{2n}\|y - X\beta\|^2 + \frac{\epsilon}{2}\|\beta\|^2 \\
   f_{sparse}(\beta) &= \|\beta\|_1 + \frac{\epsilon}{2}\|\beta\|^2 \\
   f_{smooth}(\beta) &= \frac{1+\epsilon}{2}\|\beta\|_2^2

where :math:`n` is the number of observations and :math:`\epsilon > 0` is a small constant ensuring strong convexity. These correspond to the hyperparameter mapping :math:`w = (w_1, w_2, w_3)` where:

.. math:: 
   w_1 &= \frac{1}{1 + \lambda} \\
   w_2 &= \frac{\lambda \alpha}{1 + \lambda} \\
   w_3 &= \frac{\lambda (1-\alpha)}{1 + \lambda}

By training the model on a sparse subset of weight vectors :math:`w` and fitting a Bézier simplex, we obtain a continuous **solution map** :math:`(x^*, f \circ x^*): \Delta^{M-1} \to G^*(f)` that maps any weight :math:`w` to the optimal weights :math:`\beta` and the corresponding objective values.

Model Selection on the Regularization Map
-----------------------------------------

The resulting Bézier simplex provides a continuous surrogate of the performance surface, enabling users to apply various statistical model selection criteria analytically:

1. **Minimum Cross-Validation Error (min profile rule)**: Locates the model :math:`w` that minimizes the mean cross-validation error across folds, prioritizing predictive accuracy.
2. **One Standard Error Rule (1se profile rule)**: Selects the most parsimonious (sparsest) model whose mean error is within one standard error of the minimum. This heuristic is widely used in tools like `glmnet` to gain stability and interpretability.
3. **AICc Profile Rule**: Balances goodness-of-fit and model complexity using the **Akaike Information Criterion with Finite Sample Correction**. This allows for selecting models with a theoretically grounded trade-off without the noise of fold-wise cross-validation variability.

Interactive Exploration and Insights
------------------------------------

The Bézier simplex approximation of the regularization map provides more than just a tool for optimization. It offers a **face structure** that naturally reflects the combinations of subsets of loss and regularization terms. By observing the solutions on each face of the simplex, users can:

* **Obtain Insights**: Gain a deep understanding of the trade-offs between different modeling assumptions, such as :math:`L_1` vs :math:`L_2` regularization.
* **Perform Exploratory Analysis**: Test how sensitive the optimal model is to changes in hyperparameters without the trial-and-error of retraining.
* **Support Model Selection**: Make better decisions *a posteriori* by seeing the entire landscape of potential models, rather than relying on a single fixed structure chosen *a priori*.

Empirical Validation
--------------------

The effectiveness of PyTorch-BSF for Elastic Net has been validated on various benchmark datasets from the UCI Machine Learning Repository, including:

* **Blog Feedback**: Regression task with high-dimensional features.
* **QSAR Fish Toxicity**: Predicting aquatic toxicity.
* **Slice Localization**: Estimating the relative location of CT slices.
* **Wine / Residential Building**: Predicting physical properties or prices.

Experiments show that even with a limited number of training points (e.g., 51 points for a degree-3 simplex), the Bézier simplex accurately approximates the entire solution map, maintaining low Mean Squared Error (MSE) across the continuous hyperparameter space.
