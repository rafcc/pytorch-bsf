Data Normalization
==================

Proper normalization of your data is crucial for stable and accurate Bézier simplex fitting, as both the input parameters and the output values should ideally be within a well-defined range.

Parameter Normalization
-----------------------

The ``fit()`` function expects each row of the ``params`` tensor to lie on the standard simplex :math:`\Delta^{M-1}` (i.e., the elements must be non-negative and sum to 1). If your raw parameters do not satisfy this condition, you must normalize them manually before fitting.

A common approach is **L1 Normalization**: divide each parameter vector by its :math:`\ell_1` norm. For parameters :math:`a, b \ge 0`, you can compute the simplex coordinates as :math:`t_1 = a / (a + b)` and :math:`t_2 = b / (a + b)`.

While the best normalization method depends on your specific problem, the L1 approach is a solid default. If your parameter space does not naturally have a simplex structure and no intuitive normalization seems appropriate, it is possible that the problem is not well-suited for Bézier simplex fitting.

Value Normalization
-------------------

Normalizing the output ``values`` (the targets) can help the optimizer converge faster by ensuring that different objectives are on a similar scale. In the CLI or MLflow interface, you can use the ``--normalize`` argument.

Available options include:

*   ``max``: Scales values to the range ``[0, 1]`` based on the observed maximum. This is suitable when the training data covers the full expected range (upper and lower bounds) for all axes.
*   ``std``: Standardizes values to have zero mean and unit variance. This is the most common choice for general datasets.
*   ``quantile``: Transforms values based on quantiles. This is robust to outliers and is recommended when the training data contains significant extremes that might otherwise skew the scaling.

While normalization can often improve prediction accuracy and training stability, it is not a guarantee. You should select the method that best aligns with your data distribution to achieve the best results.
