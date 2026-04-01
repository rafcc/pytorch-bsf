"""torch_bsf: PyTorch implementation of Bezier simplex fitting.

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-bsf")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install)
    __version__ = "unknown"

from torch_bsf.bezier_simplex import BezierSimplex, BezierSimplexDataModule, fit, fit_kfold
__all__ = ["BezierSimplex", "BezierSimplexDataModule", "fit", "fit_kfold"]
