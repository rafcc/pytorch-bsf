Facility location and continuous approximations
=============================================

Strongly convex multi-objective frameworks naturally accommodate classic spatial and physical challenges. In Weber-type facility location problems with squared Euclidean distances :math:`f_k(x) = \|x - a_k\|^2`, the modulus-2 strong convexity of each distance objective translates directly into a simplicial Pareto set topology :cite:p:`hamada2020topology`.

Likewise, in engineering design optimizing for structural compliance or minimum energy, the objectives inherently manifest as positive-definite quadratic costs. The strong convexity ensures that descent methods, such as the multi-objective Newton's method, achieve superlinear or even quadratic local convergence along the Pareto front :cite:p:`fliege2009newton`. 

These properties are explicitly leveraged by Adaptive Weighted Sum (AWS) methods in structural multidisciplinary optimization, where successive boundary-constrained strongly-convex QPs smoothly and uniformly map the Pareto front :cite:p:`deweck2004adaptive`. Even in complex biological systems, such as competitive Lotka-Volterra models, transformations of the quadratic interaction terms yield strongly convex models that exhibit identically stable simplicial Pareto behavior.
