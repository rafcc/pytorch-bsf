Initial Control Points
======================

You can provide initial control points with a file.
The file should contain a list of control points.
The file format should be pickled pytorch (``.pt``), comma-separated values (``.csv``), tab-separated values (``.tsv``), JSON (``.json``), or YAML (``.yml`` or ``.yaml``).

Pickled PyTorch
---------------

See PyTorch documentation for details:

- https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
- https://pytorch.org/docs/stable/generated/torch.load.html#torch.load

CSV
---

.. code-block::

   "(2, 0)", 0.0, 0.1
   "(1, 1)", 1.0, 1.1
   "(0, 2)", 2.0, 2.1

TSV
---

.. code-block::

   "(2, 0)"	0.0	0.1
   "(1, 1)"	1.0	1.1
   "(0, 2)"	2.0	2.1

JSON
----

.. code-block:: json

   {
     "(2, 0)": [0.0, 0.1],
     "(1, 1)": [1.0, 1.1],
     "(0, 2)": [2.0, 2.1]
   }

YAML
----

.. code-block:: yaml

   (2, 0): [0.0, 0.1]
   (1, 1): [1.0, 1.1]
   (0, 2): [2.0, 2.1]
