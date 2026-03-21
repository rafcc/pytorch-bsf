Robust portfolio management
===========================

In financial engineering, the most classical real-world application is Markowitz's mean-variance portfolio optimization, where investors seek to simultaneously maximize expected returns and minimize risk (variance of return). 
In practice, estimating the covariance matrix from limited observations often leads to numerical instability. To resolve this, it is common to introduce a strongly convex regularization term, such as an :math:`L_2` norm penalty, on the asset allocation weights :cite:p:`qi2026optimal`.

This formulation effectively creates a three-objective optimization problem: 

.. math::

   \text{Expected Return: } & f_1(w) = -\mu^T w \\
   \text{Risk (Variance): } & f_2(w) = w^T \Sigma w \\
   \text{Stability (Regularization): } & f_3(w) = \lambda \|w\|_2^2

Subject to the classical constraints :math:`\mathbf{1}^T w = 1` and :math:`w \ge 0` :cite:p:`markowitz1952portfolio`.

Because the regularization term is strongly convex, the resulting scalarized objective function becomes strictly strongly convex. According to the theorems in :cite:p:`mizota2021unconstrained`, this strongly convex problem is guaranteed to be weakly simplicial. Fitting a Bézier simplex to this problem allows practitioners to continuously map the entire robust Pareto front—namely, the efficient frontier—guaranteeing unique solutions. Utilizing sensitivity-based Newton path-following, the full Pareto front can be computed with an extremely efficient iteration complexity of :math:`O(p \log(1/\varepsilon))` :cite:p:`bergou2021complexity`. 
Solvers like MOSEK, Gurobi, and CVXPY are widely used to efficiently sweep across these parameters. In real-world deployments such as institutional pension fund management (e.g., GPIF), these frameworks accommodate budget limits, long-only constraints, and turnover restrictions. When cardinality constraints are added, the problem escalates to a Mixed-Integer Quadratic Program (MIQP), but the continuous relaxation remains strongly convex, allowing dedicated solvers like OSQP to execute high-frequency rebalancing robustly. Modern extensions also include CVaR (Conditional Value-at-Risk) portfolio optimization and robust multi-objective portfolio frameworks under data uncertainty.
