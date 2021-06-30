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
Let :math:`\mathbb N` be the set of nonnegative integers, :math:`D` a positive integer, and :math:`M, N` nonnegative integers.
We define the *index set* by

.. math::\mathbb N_D^M = \left\{(d_1,\ldots,d_M)\in\mathbb N^M | \sum_{m=1}^M d_m=D\right\},

and the :math:`(M-1)`*-simplex* by

.. math::\Delta^{M-1} = \left\{(t_1,\ldots,t_M)\in\mathbb R^M | \sum_{m=1}^M t_m=1\right\}.

An :math:`(M-1)`-dimensional *Bezier simplex* of degree :math:`D` in :math:`\mathbb R^N` is a polynomial map :math:`b: \Delta^{M-1}\to\mathbb R^N` defined by

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