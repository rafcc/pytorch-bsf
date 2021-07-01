What is Bezier simplex fitting?
===============================

Bezier simplex
--------------
Let :math:`D, M, N` be nonnegative integers, :math:`\mathbb N` the set of nonnegative integers, and :math:`\mathbb R^N` the :math:`N`-dimensional Euclidean space.
We define the *index set* by

.. math:: \mathbb N_D^M = \left\{(d_1,\ldots,d_M)\in\mathbb N^M \Big| \sum_{m=1}^M d_m=D\right\},

and the *simplex* by

.. math:: \Delta^{M-1} = \left\{(t_1,\ldots,t_M)\in\mathbb R^M \Big| \sum_{m=1}^M t_m=1\right\}.

An :math:`(M-1)`-dimensional *Bezier simplex* of degree :math:`D` in :math:`\mathbb R^N` is a polynomial map :math:`b: \Delta^{M-1}\to\mathbb R^N` defined by

.. math:: b(t|p) = \sum_{d\in\mathbb N_D^M} \binom{D}{d} t^d p_d,

where :math:`t^d=t_1^{d_1} t_2^{d_2}\cdots t_M^{d_M}`, :math:`\binom{D}{d}=D! / (d_1!d_2!\cdots d_M!)`, and :math:`p_d\in\mathbb R^N\ (d\in\mathbb N_D^M)` are parameters called the *control points*.


Fitting a Bezier simplex to a dataset
-------------------------------------
Assume we have a finite dataset :math:`B\subset\Delta^{M-1}\times\mathbb R^N` and want to fit a Bezier simplex to the dataset.
What we are trying can be formulated as a problem of finding the best vector of control points :math:`p=(p_d)_{d\in\mathbb N_D^M}` that minimizes the least square error between the Bezier simplex and the dataset:

.. math:: \min_{p} \sum_{(t,x)\in B}\|b(t|p)-x\|^2.

PyTorch-BSF provides an algorithm for solving this optimization problem with the L-BFGS algorithm.


Why does Bezier simplex fitting matter?
---------------------------------------
The Bezier simplex can approximate the solution set of "good" multiobjective optimization problems.
More precisely, for the weighted sum scalarization problem of any multiobjective strongly convex problem, the map from a simplex of weight vectors to the solution set of weighted sum problems can be approximated by a Bezier simplex.
If we find few solutions to such a problem, the entire solution set can be approximated by Bezier simplex fitting.
An important application is hyperparameter search of the elastic net.
