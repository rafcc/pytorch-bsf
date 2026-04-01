Advanced Sampling
=================

PyTorch-BSF provides utilities for generating parameter points on a simplex beyond the default uniform grid.

The ``torch_bsf.sampling`` module
----------------------------------

Three functions are available:

``simplex_grid(n_params, degree)``
    Generates a uniform grid on the simplex using the stars-and-bars method.
    This is the default sampling used by ``fit()`` and the CLI.

``simplex_random(n_params, n_samples)``
    Generates random points uniformly distributed over the simplex via the Dirichlet distribution.
    Useful when you want stochastic coverage or a large number of samples.

.. code-block:: python

   from torch_bsf.sampling import simplex_random

   pts = simplex_random(n_params=3, n_samples=200)

``simplex_sobol(n_params, n_samples)``
    Generates quasi-random points using a scrambled Sobol sequence projected onto the simplex.
    Sobol sequences provide better coverage than purely random sampling, reducing clustering and gaps.
    Requires ``scipy`` (``pip install scipy``).

.. code-block:: python

   from torch_bsf.sampling import simplex_sobol

   pts = simplex_sobol(n_params=3, n_samples=200)

When to Use Each Method
-----------------------

+-------------------+----------------------------------------------------------+
| Method            | Recommended use                                          |
+===================+==========================================================+
| ``simplex_grid``  | Small, structured datasets; default CLI behavior.        |
+-------------------+----------------------------------------------------------+
| ``simplex_random``| Large-scale or stochastic sampling experiments.          |
+-------------------+----------------------------------------------------------+
| ``simplex_sobol`` | When uniform coverage is critical (e.g., active learning |
|                   | initialization, quasi-Monte Carlo integration).          |
+-------------------+----------------------------------------------------------+
