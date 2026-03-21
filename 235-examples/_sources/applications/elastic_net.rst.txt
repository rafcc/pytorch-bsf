Elastic net model selection
===========================

A canonical and highly practical application of this theory is hyperparameter optimization for the **Elastic Net**. In machine learning and statistics, selecting the correct regularization parameters is critical to balancing data fidelity, model sparsity, and numerical stability. 

By formulating optimal parameter tuning for the Elastic Net as optimization over the Pareto set of a convex multi-objective problem :cite:p:`bonnel2019post`, practitioners eliminate the need for exhaustive and computationally expensive grid searches over all hyperparameter combinations. 

The underlying multi-objective problem involves simultaneously minimizing the least-squares data fidelity, the :math:`L_1` sparsity penalty, and the :math:`L_2` smoothness penalty over the model weights :math:`\beta`:

.. math:: 
   
   f_{data}(\beta) &= \frac{1}{2}\|y - X\beta\|^2 \\
   f_{sparse}(\beta) &= \|\beta\|_1 \\
   f_{smooth}(\beta) &= \frac{1}{2}\|\beta\|_2^2

Typically, these are scalarized as :math:`\min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2`. 

Because of the :math:`L_2` penalty term, the overall objective is strongly convex. Specifically, the addition of :math:`\lambda_2\|\beta\|_2^2` ensures that the Hessian of the smooth component is lower bounded by :math:`\lambda_2 I \succ 0`. This guarantees that the underlying solution is unique for each parameter combination, enabling well-defined sensitivity analysis and path-following algorithms. Consequently, the problem is guaranteed to be weakly simplicial :cite:p:`mizota2021unconstrained`.

Rather than training thousands of models discretely, you can train the Elastic Net on a sparse subset of simplex-structured weight vectors. Fitting a Bézier simplex to the resulting trained models yields a continuous performance surface. This analytic surrogate allows practitioners to instantly explore the full continuous spectrum of model hyperparameters and locate the statistically optimal model analytically, without any further retraining.


Weighted-sum scalarization and solution map
-------------------------------------------

The *weighted-sum scalarization* :math:`x^*: \Delta^{M-1}\to\mathbb R^N` defined by

.. math:: x^*(w)=\arg\min_x \sum_{m=1}^M w_m f_m(x).

We define the *solution map* :math:`(x^*,f\circ x^*):\Delta^{M-1}\to G^*(f)` by

.. math:: (x^*,f\circ x^*)(w)=(x^*(w),f(x^*(x))).

The solution map is continuous and surjective.
See :cite:p:`mizota2021unconstrained` for technical details.
