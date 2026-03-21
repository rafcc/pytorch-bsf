Medical imaging and radiation therapy
=====================================

In the medical field, strongly convex optimization heavily supports diagnostic imaging and therapeutic planning. For Computed Tomography (CT) image reconstruction, the task is posed as an inverse problem minimizing a data fidelity term (squared error) alongside a Tikhonov-style quadratic regularization term. This is classically formulated as :math:`\min_x \{ \|Kx-y\|_2^2 + \lambda^2\|x\|_2^2 \}`. By analyzing the L-curve, practitioners sweep the scalarization weight :math:`\lambda` to balance residual and solution size :cite:p:`hansen1993use`. Even if the projection matrix is ill-posed or rank-deficient, the quadratic regularization guarantees that the objective function is strictly strongly convex (:math:`\nabla^2 f \succeq \lambda I`), permitting massive-scale voxel optimizations via proximal gradient methods or ADMM.

Similarly, Intensity-Modulated Radiation Therapy (IMRT) utilizes highly constrained multi-criteria inverse planning to determine spatial radiation doses :cite:p:`breedveld2007novel`. The continuous sub-optimizations minimize the target tumor dose errors while prioritizing Organs At Risk (OAR) limits, formulated with spatial smoothing components as:

.. math:: s(f) = \sum_{v \in V} \xi_v (Hf - d_v^p)^T \tilde{\eta}_v (Hf - d_v^p) + \kappa (Mf)^T (Mf)

Because the Hessian :math:`\nabla^2 s(f) = 2(H^T \tilde{\eta} H + \kappa M^T M)` is positive-definite, this strongly convex structure allows treatment planners to securely navigate the complex Pareto trade-offs of patient safety and tumor eradication via high-speed QP solver iterations.
