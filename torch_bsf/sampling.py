import itertools
import torch

def simplex_grid(n_params: int, degree: int) -> torch.Tensor:
    """Generates a uniform grid on a simplex.
    
    Parameters
    ----------
    n_params : int
        The number of parameters (vertices of the simplex).
    degree : int
        The degree of the grid.
        
    Returns
    -------
    torch.Tensor
        Array of grid points in shape (N, n_params), where N = binom(degree + n_params - 1, n_params - 1).
    """
    if n_params <= 0:
        return torch.empty((0, 0))
    if degree == 0:
        return torch.ones((1, n_params), dtype=torch.float32) / n_params
    
    # Stars and bars method to generate all non-negative integer solutions to sum(i_j) = degree
    # Ps contains all picking positions for 'bars' in a row of degree + n_params - 1 positions.
    ps = torch.tensor(list(itertools.combinations(range(degree + n_params - 1), n_params - 1)))
    
    # Use vectorized differences to calculate the counts (stars) in each gap
    N = ps.shape[0]
    padded = torch.zeros((N, n_params + 1), dtype=torch.long)
    padded[:, 1:-1] = ps + 1
    padded[:, -1] = degree + n_params
    
    diffs = padded[:, 1:] - padded[:, :-1] - 1
    
    return diffs.float() / degree


