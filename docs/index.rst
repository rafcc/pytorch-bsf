.. PyTorch-BSF documentation master file, created by
   sphinx-quickstart on Sun Jun 27 02:31:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch-BSF's documentation!
=======================================
PyTorch-BSF is a PyTorch implementation of Bezier simplex ftting.
The project is on `GitHub`_.

Bezier simplex
==============
Let $\mathbb N$ be the set of nonnegative integers.
An :math:`(M-1)`-dimensional Bezier simplex of degree :math:`D` in :math:`\mathbb R^N` is a map :math:`b: \Delta^{M-1}\to\mathbb R^N` defined by

.. math::
   b(t) = \sum_{d\in\mathbb N_D^M} \binom{D}{d} t^d p_d,

where :math:`p_d\in\mathbb R^N\ (d\in\mathbb N_D^M)` are parameters called the *control points*.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/rafcc/pytorch-bst