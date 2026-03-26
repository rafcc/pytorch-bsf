Automatic Degree Selection
==========================

Choosing the optimal degree for a Bézier simplex can be challenging. A degree that is too low may underfit and fail to capture the manifold's complexity, while a degree that is too high may overfit to noise. 

PyTorch-BSF provides an automated tool to select the best degree based on **k-fold cross-validation**.

How to Use
----------

The ``select_degree()`` function iterates through multiple degrees and evaluates the mean squared error (MSE) using cross-validation.

.. code-block:: python

   import torch_bsf
   from torch_bsf.model_selection.degree_selection import select_degree

   # Your data
   ts = ... # parameters
   xs = ... # values

   # Select the best degree between 1 and 5
   best_d = select_degree(
       params=ts, 
       values=xs, 
       min_degree=1, 
       max_degree=5, 
       num_folds=5, 
       max_epochs=2,
       accelerator="auto" # use GPU if available
   )

   # Use the best degree for final fitting
   bs = torch_bsf.fit(params=ts, values=xs, degree=best_d)

How it Works
------------

1.  **Iteration**: The tool trains Bézier simplices for degrees from ``min_degree`` to ``max_degree``.
2.  **K-Fold CV**: For each degree, it performs :math:`k`-fold cross-validation. The mean MSE across all validation folds is calculated.
3.  **Heuristic Search**: If the CV error increases for a degree compared to the previous one, the search stops early to save time.
4.  **Optimal Selection**: The degree with the lowest mean validation MSE is returned.

Configuration
-------------

The ``select_degree()`` function accepts any keyword arguments that the standard PyTorch Lightning `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_ supports. 

Important parameters include:

*   ``min_degree`` / ``max_degree``: Range of degrees to check.
*   ``num_folds``: Number of cross-validation folds.
*   ``max_epochs``: Number of epochs to train for each fold (smaller is faster, but might be less accurate for selection).
*   ``accelerator`` / ``devices``: Hardware settings for training.

When to Use
-----------

1.  **Exploratory Data Analysis**: When you are unsure about the underlying manifold's complexity.
2.  **Automated Processes**: If you are building an automated pipeline for Pareto front approximation.
3.  **Model Optimization**: When accuracy is more important than training time, and you want the most precisely tuned model.
