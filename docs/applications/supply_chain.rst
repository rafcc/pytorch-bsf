Supply chain and logistics optimization
=======================================

In logistics and supply chain management (SCM), integrating physical material flows with monetary information (Supply Chain Finance) is critical for maximizing overall profitability. Recent formulations model this as a constrained finite-horizon Linear Quadratic Regulator (LQR) problem, solving it as a convex Quadratic Program (QP) :cite:p:`mdpi_sc_finance`. 

The optimization simultaneously minimizes three strongly convex objectives:

1. Deviation from target profit/cash states.
2. Excessive external financing or operational inputs.
3. Abrupt policy changes (operational stability).

Because the overarching LQR formulation is inherently governed by positive definite penalty matrices, the entire supply chain network optimization is globally strongly convex. This permits off-the-shelf interior-point solvers to efficiently resolve optimal strategies across complex multi-node production flows.
