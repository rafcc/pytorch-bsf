.. PyTorch-BSF documentation master file, created by
   sphinx-quickstart on Sun Jun 27 02:31:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quickstart
==========
There are several ways to use PyTorch-BSF.


Run as an MLflow project
------------------------
The fastest way is to use MLflow projects.
Assume you have installed Miniconda.
Install the conda package ``mlflow`` from the channel ``conda-forge``:

.. code-block:: bash

   conda install -c conda-forge mlflow


Now, you can fit a Bezier simplex to a dataset with the latest version:

.. code-block:: bash

   mlflow run https://github.com/rafcc/pytorch-bsf \
   -P data=data.tsv \
   -P label=label.tsv \
   -P degree=3

.. code-block:: bash

   mlflow models serve <run_id>


Run as a Python package
-----------------------
Assume you have installed Python 3.8 or above.
Then, install the package:

.. code-block:: bash

  pip install pytorch-bsf


Run as Python code
------------------
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
