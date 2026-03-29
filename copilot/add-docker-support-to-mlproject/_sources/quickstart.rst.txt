Quickstart
==========

PyTorch-BSF can be used in three ways depending on your workflow: as a zero-install **MLflow project** (great for one-off experiments), as a **CLI module** (scriptable, no Python required), or as a **Python library** (for full programmatic control).
Pick the option that best fits your setup.


Run as an MLflow project
------------------------

If you have parameters and values for training a Bézier simplex in common file formats such as CSV, JSON, etc., then the easiest way is to invoke PyTorch-BSF via `MLflow`_.
In this way, some CLI commands for training and prediction are provided without installing PyTorch-BSF.
On each training and prediction, separation of runtime environment and installation of PyTorch-BSF are automatically handled by MLflow!

.. _MLflow: https://www.mlflow.org/docs/latest/


Installation
^^^^^^^^^^^^

Choose Docker (the default) or Conda as the environment manager.

**Docker (default)**

Install `Docker`_ and then install ``mlflow`` via pip:

.. code-block:: bash

   pip install mlflow

MLflow will pull a pre-built image from GHCR that installs PyTorch via the ``pytorch`` conda channel, providing Intel MKL as the BLAS backend.

.. _Docker: https://docs.docker.com/get-docker/

**Conda**

Install `Miniconda`_ (or Anaconda) and then install ``mlflow`` via pip:

.. code-block:: bash

   pip install mlflow

MLflow will create a conda environment from the project's ``environment.yml``, which also uses the ``pytorch`` conda channel and Intel MKL.

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html


Training
^^^^^^^^

Let's prepare sample parameters and values files for training:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:CreateFiles]
   :end-before: [TAG:CreateFiles_End]

.. warning::
   The parameters file and the values file must have the same number of lines.

Now, you can fit a Bézier simplex model using the latest version of PyTorch-BSF directly from its GitHub repository.

**Docker (default)** — pulls a GHCR image with MKL-backed PyTorch:

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf \
     -P params=params.csv \
     -P values=values.csv \
     -P degree=3

**Conda** — creates a conda environment with MKL-backed PyTorch:

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf \
     --env-manager=conda \
     -P params=params.csv \
     -P values=values.csv \
     -P degree=3

After the command finishes, the trained model will be saved in ``mlruns/0`` directory.
Note the **Run ID** automatically set to the command execution, as you will need it for prediction.


Prediction
^^^^^^^^^^

To make predictions, MLflow may use ``virtualenv`` and ``pyenv`` to create an isolated environment for the model. Please ensure it's available in your system.

First, find the **Run ID** (e.g., `47a7...`) from the previous training step.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:FetchLatestRunID]
   :end-before: [TAG:FetchLatestRunID_End]


Next, you can predict with the model and output the results to a specified file (in this example, `test_values.json`).

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:MakePrediction]
   :end-before: [TAG:MakePrediction_End]

See https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-models-predict for details.


Serve prediction API
^^^^^^^^^^^^^^^^^^^^

You can also serve a Web API for prediction.

First, find the Run ID (e.g., `a1b2c3...`) set to the model training.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:FetchLatestRunID]
   :end-before: [TAG:FetchLatestRunID_End]


Then, start a prediction server using the Run ID.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:ServeAPI]
   :end-before: [TAG:ServeAPI_End]


Now, you can request a prediction with HTTP POST method:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:PredictWithHTTPPost]
   :end-before: [TAG:PredictWithHTTPPost_End]


See https://mlflow.org/docs/latest/genai/serving/ for details.


Run as a Python package
-----------------------

Assume you have installed Python 3.10 or above.
Then, install the package:

.. code-block:: bash

   pip install pytorch-bsf

Then, run `torch_bsf` as a module:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:RunPackageTraining]
   :end-before: [TAG:RunPackageTraining_End]


Run as Python code
------------------

Assume you have installed Python 3.10 or above.
Then, install the package:

.. code-block:: bash

   pip install pytorch-bsf

Train a model by ``fit()``, and call the model to predict.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf

   # Prepare training parameters
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

   # Predict with the trained model
   t = [
      [0.2, 0.8],
      [0.7, 0.3],
   ]
   x = bs(t)
   print(x)

.. testoutput::
   :hide:

   tensor([[...]], grad_fn=<...>)
