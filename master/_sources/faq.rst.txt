Frequently asked questions
==========================

TBA.


There are too many tools for hyperparameter search of ML. Why do you propose yet another one?
---------------------------------------------------------------------------------------------

There are many important differences between the PyTorch-BSF and other hyperparameter search tools.
First, the domain of a Bezier simplex is a (bounded) simplex, whereas other tools assume a (non-bounded) Euclidean space.
Thus, the Bezier simplex can represent a compact subspace of the ambient space.

Second, for hyperparameter search tasks, the Bezier simplex has a strong limitation that its only optimizable type of hyperparameters is coefficients in the objective function.
However, this limitation offers a great advantage in speed.
While black-box optimization algorithms do not use properties of the objective function, the Bezier simplex takes advantage of it.
As a result, the Bezier simplex fitting requires fewer solutions to fit the entire Pareto set and front, which will be suitable for hyperparameter search.
And also you will find such a type of hyperparameters is quite ubiquitous.
In designing software, there is always a trade-off between generality and efficiency.


Are approximation results always reliable?
------------------------------------------------

No, not always.
The approximation theorem says nothing about the approximation accuracy of the Bezier simplex of **fixed** degree.
After fitting a Bezier simplex, you need to check the goodness of fit by using your domain knowledge.


Are there any applications other than multiobjective optimization?
------------------------------------------------------------------

Not yet, but possibly yes.
I believe Bezier simplices arise everywhere we need high-dimensional shape representation.
If you find a new application, then please let me know it.
Your application will be documented in the application section.