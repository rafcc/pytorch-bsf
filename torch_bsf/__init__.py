"""torch_bsf: PyTorch implementation of Bezier simplex fitting.

"""

from torch_bsf._version import __version__
from torch_bsf.bezier_simplex import BezierSimplex, BezierSimplexDataModule, fit
__all__ = ["BezierSimplex", "BezierSimplexDataModule", "fit"]
