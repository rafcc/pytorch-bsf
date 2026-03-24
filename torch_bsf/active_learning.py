from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from torch_bsf.bezier_simplex import BezierSimplex
from torch_bsf.sampling import simplex_grid, simplex_random


def suggest_next_points(
    models: List[BezierSimplex],
    n_suggestions: int = 1,
    n_candidates: int = 1000,
    method: str = "qbc",
    params: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Suggest points on the simplex where new data should be sampled.

    Parameters
    ----------
    models : List[BezierSimplex]
        An ensemble of models (e.g., from k-fold cross-validation).
    n_suggestions : int, default=1
        The number of points to suggest.
    n_candidates : int, default=1000
        The number of candidate points to evaluate.
    method : str, default="qbc"
        The method to use:
        - "qbc": Query-By-Committee. Suggests points where models disagree most.
        - "density": Suggests points that are furthest from existing training points.
    params : torch.Tensor, optional
        The existing training parameters. Required for method="density".

    Returns
    -------
    torch.Tensor
        The suggested points in shape (n_suggestions, n_params).
    """
    n_params = models[0].n_params
    device = models[0].device

    # Generate candidate points
    candidates = simplex_random(n_params, n_candidates).to(device)

    if method == "qbc":
        # Calculate variance across models
        preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                preds.append(model(candidates))
        
        # preds: (K, n_candidates, n_values)
        preds = torch.stack(preds)
        
        # Variance of predictions: (n_candidates, n_values)
        var = torch.var(preds, dim=0)
        
        # Sum of variance across dimensions: (n_candidates,)
        uncertainty = torch.sum(var, dim=1)
        
        # Select top candidates
        _, indices = torch.topk(uncertainty, n_suggestions)
        return candidates[indices]

    elif method == "density":
        if params is None:
            raise ValueError("params must be provided for method='density'")
        
        # Calculate distance to nearest neighbor in parameter space
        # candidates: (C, M), params: (N, M)
        # We use squared Euclidean distance
        dist = torch.cdist(candidates, params.to(device))
        
        # min distance to any existing point: (C,)
        min_dist, _ = torch.min(dist, dim=1)
        
        # Select points with largest min_dist
        _, indices = torch.topk(min_dist, n_suggestions)
        return candidates[indices]

    else:
        raise ValueError(f"Unknown method: {method}")
