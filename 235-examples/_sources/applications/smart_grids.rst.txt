Distributed smart grids
=======================

In modern energy management systems, the Multi-Objective Optimal Power Flow (MO-OPF) problem aims to minimize operating costs while precisely matching power supply and demand. For traditional generators, the fuel cost is typically formulated as a quadratic function of power output :math:`P_i`, which is inherently strongly convex:

.. math:: C_i(P_i) = \alpha_i P_i^2 + \beta_i P_i + \gamma_i \quad (\alpha_i > 0)

Moreover, environmental constraints such as minimizing emissions (NOx/SOx) and minimizing transmission losses (using B-coefficient matrices) transform this into an Economic Emission Dispatch (EED) multi-objective problem, where all core objectives are quadratic and strongly convex.
Because the objectives are strongly convex, advanced decentralized frameworks (like distributed Lagrangian methods or ADMM) can achieve linear convergence across the geographic network :cite:p:`performing2025accelerated,multi2026economic`. For instance, in multi-agent microgrid networks, predefined-time distributed algorithms exploit this strong convexity and smoothness to achieve exponential convergence to the Pareto optimum within a strict time horizon, utilizing Zeno-free event-triggered communication :cite:p:`zhang2023predefined`. This robust mathematical property guarantees that even in unstable communication environments with latency or differential privacy additions, the network can securely and optimally coordinate energy distribution. 

Furthermore, in integrated photovoltaic and battery EV charging stations, the demand mismatch penalty is commonly modeled as a strongly convex quadratic utility function :math:`u_l = -\frac{1}{2}(p_l - p_l^*)^2`, enabling multi-agent game-theoretic frameworks to rapidly coordinate distributed power allocation via Nash equilibria :cite:p:`yan2021two`. Similar principles apply to microgrid energy management, balancing generator fuel costs against battery degradation.
