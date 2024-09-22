"""torch_bsf.model_selection: PyTorch implementation of Bezier simplex fitting.

This module provides methods for model selection.
"""
from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid

__all__ = ["elastic_net_grid"]
