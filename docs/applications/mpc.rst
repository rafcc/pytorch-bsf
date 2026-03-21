Multi-objective model predictive control
========================================

Autonomous systems like self-driving cars, drones, and industrial robots rely on Model Predictive Control (MPC) and Linear Quadratic Regulators (LQR) to optimize future behavior in real-time. In complex scenarios, these systems face competing goals such as precise trajectory tracking, energy efficiency (minimizing control effort/power), and motion smoothness (minimizing jerk). When these objectives are framed as positive-definite quadratic forms, the resulting stage cost function is strictly strongly convex:

.. math:: l(x, u) = x^T Q x + u^T R u

Where :math:`Q` and :math:`R` are positive definite matrices. Even though the set of stabilizing controllers might be non-convex, the multi-objective LQR's Pareto front can be completely characterized by linear scalarization due to these strongly convex quadratic structures. This strong convexity guarantees that the underlying Quadratic Programming (QP) problem has a unique optimal solution that can be solved extremely rapidly at each time step. Furthermore, it inherently ensures the stability, safety (recursive feasibility), and collision-avoidance of distributed multi-agent formations (like drone swarms), acting as a stabilizing anchor in highly uncertain physical environments :cite:p:`distributed2022model,multi2024learning`.

These principles are widely deployed in the chemical process industry via Predictive Functional Control (PFC), where plant operators balance quality tracking with the suppression of abrupt heater/valve operations :cite:p:`sice_process_mpc`. Similarly, in robotics, the continuous trajectory generation for omni-directional warehouse robots is modeled as a strongly convex QP. Factoring in discrete decisions like friction mode switching transforms the control task into an MIQP, yet the strong convexity of the continuous domain dramatically stabilizes the branch-and-bound optimization nodes :cite:p:`jsme_robotics_miqp`.
