Advanced Topics
===============

Advanced topics for more customized training.


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

.. code-block:: csv

   "(2, 0)", 0.0, 0.1, 0.2
   "(1, 1)", 1.0, 1.1, 1.2
   "(0, 2)", 2.0, 2.1, 2.2


TSV
^^^

.. code-block:: tsv

   "(2, 0)"	0.0	0.1	0.2
   "(1, 1)"	1.0	1.1	1.2
   "(0, 2)"	2.0	2.1	2.2


JSON
^^^^

.. code-block:: json

   {
     "(2, 0)": [0.0, 0.1, 0.2],
     "(1, 1)": [1.0, 1.1, 1.2],
     "(0, 2)": [2.0, 2.1, 2.2]
   }


YAML
^^^^

.. code-block:: yaml

   "(2, 0)": [0.0, 0.1, 0.2],
   "(1, 1)": [1.0, 1.1, 1.2],
   "(0, 2)": [2.0, 2.1, 2.2]


Partial training
----------------

Funciton ``fit()`` accepts partial training, i.e., train some of control points while the others are fixed.

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

   # Train the edges and surface of a Bezier triangle while its vertices are fixed
   bs = torch_bsf.fit(
      params=ts,  # input observations (training data)
      values=xs,  # output observations (training data)
      init="control_points.yml",  # initial values of control points
      fix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # fix vertices of the Bezier triangle
   )

   # Predict by the trained model
   t = [[0.2, 0.3, 0.5]]
   x = bs(t)
   print(f"{t} -> {x}")
