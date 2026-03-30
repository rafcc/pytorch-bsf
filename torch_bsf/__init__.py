"""torch_bsf: PyTorch implementation of Bezier simplex fitting.

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-bsf")
except PackageNotFoundError:
    __version__ = "unknown"

from torch_bsf.bezier_simplex import BezierSimplex, BezierSimplexDataModule, fit
__all__ = ["BezierSimplex", "BezierSimplexDataModule", "fit"]
