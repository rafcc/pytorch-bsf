Quickstart
==========

There are several ways to use PyTorch-BSF.


Run as an MLflow project
------------------------

If you have data and labels for training a Bezier simplex in common file formats such as CSV, JSON, etc., then the easiest way is to invoke PyTorch-BSF via `MLflow`_.
In this way, some CUI commands for training and prediction are provided without installing PyTorch-BSF.
On each training and prediction, separation of runtime environment and installation of PyTorch-BSF are automatically handled by MLflow!

.. _MLflow: https://www.mlflow.org/docs/latest/


Installation
^^^^^^^^^^^^

First, install `Miniconda`_.
Then, install ``mlflow`` conda package from ``conda-forge`` channel:

.. code-block:: bash

   conda install -c conda-forge mlflow

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html


Training
^^^^^^^^

Let's prepare data and labels for training:

.. code-block:: bash

   cat << EOS > params.csv
   1.00, 0.00
   0.75, 0.25
   0.50, 0.50
   0.25, 0.75
   0.00, 1.00
   EOS
   cat <<EOS > values.csv
   0.00, 1.00
   3.00, 2.00
   4.00, 5.00
   7.00, 6.00
   8.00, 9.00
   EOS

.. warning::
   The data file and label file must have the same number of lines.

Now, you can fit a Bezier simplex to those data and labels with the latest version of PyTorch-BSF:

.. code-block:: bash

   mlflow run https://github.com/rafcc/pytorch-bsf \
   -P params=params.csv \
   -P values=values.csv \
   -P meshgrid=params.csv \
   -P degree=3


After the command finished, you will get a trained model in ``mlruns`` directory.


Prediction
^^^^^^^^^^

.. code-block:: bash

   mlflow models predict \
     --model-uri file://`pwd`/mlruns/0/${run_uuid}/artifacts/model \
     --content-type csv \
     --input-path params.csv \
     --output-path test_values.csv


You have results in ``test_values.csv``:

.. code-block:: bash

   cat test_values.csv
   [[0.00, 1.00], ...]


Serve prediction API
^^^^^^^^^^^^^^^^^^^^

You can also serve a Web API for prediction:

.. code-block:: bash

   mlflow models serve \
     --model-uri {Full Path} \
     --host localhost \
     --port 5001


Request a prediction with HTTP POST method:

.. code-block:: bash

   curl http://localhost:5001/invocations -H 'Content-Type: application/json' -d '{
     "columns": ["t1", "t2"],
     "data": [
        [0.2, 0.8],
        [0.7, 0.3]
     ]
   }'

See for details https://www.mlflow.org/docs/latest/models.html#deploy-mlflow-models


Run as a Python package
-----------------------

Assume you have installed Python 3.8 or above.
Then, install the package:

.. code-block:: bash

  pip install pytorch-bsf

Then, run `torch_bsf` as a module:

.. code-block:: bash

   python -m torch_bsf \
     --model-uri file://`pwd`/mlruns/0/${run_uuid}/artifacts/model \
     --content-type csv \
     --input-path test_params.csv \
     --output-path test_values.csv


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
         [8/8, 0/8],
         [7/8, 1/8],
         [6/8, 2/8],
         [5/8, 3/8],
         [4/8, 4/8],
         [3/8, 5/8],
         [2/8, 6/8],
         [1/8, 7/8],
         [0/8, 8/8],
      ]
   )
   xs = 1 - ts * ts  # values corresponding to the parameters

   # Train a model
   bs = torch_bsf.fit(params=ts, values=xs, degree=3)

   # Predict by the trained model
   t = [
      [0.2, 0.8],
      [0.7, 0.3],
   ]
   x = bs(t)
   print(f"{t} -> {x}")
