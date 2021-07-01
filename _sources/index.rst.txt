.. PyTorch-BSF documentation master file, created by
   sphinx-quickstart on Sun Jun 27 02:31:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch-BSF!
=======================
PyTorch-BSF is a PyTorch implementation of Bezier simplex ftting.
The project is on `GitHub`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   whatis
   modules


Quickstart
==========
Assume you have installed Python 3.8 or above.
Then, install the package:

.. code-block:: bash

  pip install pytorch-bsf

Train a model by ``fit()``, and call the model to predict.

.. code-block:: python

   import torch
   import torch_bsf

   # Prepare training data
   ts = torch.tensor(  # parameters on a simplex
      [
         [3/3, 0/3, 0/3],
         [2/3, 1/3, 0/3],
         [2/3, 0/3, 1/3],
         [1/3, 2/3, 0/3],
         [1/3, 1/3, 1/3],
         [1/3, 0/3, 2/3],
         [0/3, 3/3, 0/3],
         [0/3, 2/3, 1/3],
         [0/3, 1/3, 2/3],
         [0/3, 0/3, 3/3],
      ]
   )
   xs = 1 - ts * ts  # values corresponding to the parameters

   # Train a model
   bs = torch_bsf.fit(params=ts, values=xs, degree=3, max_epochs=100)

   # Predict by the trained model
   t = [[0.2, 0.3, 0.5]]
   x = bs(t)
   print(f"{t} -> {x}")


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/rafcc/pytorch-bsf