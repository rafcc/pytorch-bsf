Advanced Topics
===============

This section covers customization options beyond the basic ``fit()`` call: loading pre-computed control points as an initialization, pinning specific control points during training, and more.
These features are useful when you have prior knowledge about the shape of the Bézier simplex or want to perform incremental refinement.


Initial control points
----------------------

You can provide initial control points with a file.
The file should contain a list of control points.
The file format should be pickled pytorch (``.pt``), comma-separated values (``.csv``), tab-separated values (``.tsv``), JSON (``.json``), or YAML (``.yml`` or ``.yaml``).


Pickled PyTorch
^^^^^^^^^^^^^^^

See PyTorch documentation for details:

- https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
- https://pytorch.org/docs/stable/generated/torch.load.html#torch.load


CSV
^^^

.. code-block::

   "(2, 0)", 0.0, 0.1
   "(1, 1)", 1.0, 1.1
   "(0, 2)", 2.0, 2.1


TSV
^^^

.. code-block::

   "(2, 0)"	0.0	0.1
   "(1, 1)"	1.0	1.1
   "(0, 2)"	2.0	2.1


JSON
^^^^

.. code-block:: json

   {
     "(2, 0)": [0.0, 0.1],
     "(1, 1)": [1.0, 1.1],
     "(0, 2)": [2.0, 2.1]
   }


YAML
^^^^

.. code-block:: yaml

   (2, 0): [0.0, 0.1]
   (1, 1): [1.0, 1.1]
   (0, 2): [2.0, 2.1]


Partial training
----------------

Function ``fit()`` provides some arguments for partial training, i.e., train some of control points while the others are fixed.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf

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

   # Initialize 2D control points of a Bezier triangle of degree 3
   init = {
      # index: value
      (3, 0): [0.0, 0.1],
      (2, 1): [1.0, 1.1],
      (1, 2): [2.0, 2.1],
      (0, 3): [3.0, 3.1],
   }

   # Or, generate random control points in [0, 1)
   init = torch_bsf.bezier_simplex.rand(n_params=2, n_values=2, degree=3)

   # Or, load control points from a file
   init = torch_bsf.bezier_simplex.load("control_points.yml")

   # Train the edge of a Bezier curve while its vertices are fixed
   bs = torch_bsf.fit(
      params=ts,  # input observations (training data)
      values=xs,  # output observations (training data)
      init=init,  # initial values of control points
      fix=[[3, 0], [0, 3]],  # fix vertices of the Bezier curve
   )

   # Predict by the trained model
   t = [
      [0.2, 0.8],
      [0.7, 0.3],
   ]
   x = bs(t)
   print(x)

.. testoutput::
   :hide:

   tensor([[...]], grad_fn=<AddBackward0>)
