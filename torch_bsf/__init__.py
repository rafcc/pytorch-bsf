"""torch_bsf: PyTorch implementation of Bezier simplex fitting.

"""

from torch_bsf.bezier_simplex import BezierSimplex, BezierSimplexDataModule, fit

__version__ = "0.13.1"
__all__ = ["BezierSimplex", "BezierSimplexDataModule", "fit"]
