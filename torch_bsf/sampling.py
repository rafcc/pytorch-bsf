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
    ps = torch.tensor(
        list(itertools.combinations(range(degree + n_params - 1), n_params - 1))
    )

    # Use vectorized differences to calculate the counts (stars) in each gap
    N = ps.shape[0]
    padded = torch.zeros((N, n_params + 1), dtype=torch.long)
    padded[:, 1:-1] = ps + 1
    padded[:, -1] = degree + n_params

    diffs = padded[:, 1:] - padded[:, :-1] - 1

    return diffs.float() / degree


def simplex_random(n_params: int, n_samples: int) -> torch.Tensor:
    """Generates random points on a simplex using Dirichlet distribution.

    Parameters
    ----------
    n_params : int
        The number of parameters (vertices of the simplex).
    n_samples : int
        The number of samples.

    Returns
    -------
    torch.Tensor
        Array of sample points in shape (n_samples, n_params).

    Raises
    ------
    ValueError
        If ``n_params`` is not positive or ``n_samples`` is negative.
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_samples == 0:
        return torch.empty((0, n_params), dtype=torch.float32)

    import numpy as np

    # Sample from Dirichlet distribution with concentration = ones
    # (Uniform distribution over the simplex)
    samples = np.random.dirichlet([1.0] * n_params, n_samples)
    return torch.from_numpy(samples).float()


def simplex_sobol(n_params: int, n_samples: int) -> torch.Tensor:
    """Generates quasi-random points on a simplex using Sobol sequence.

    Parameters
    ----------
    n_params : int
        The number of parameters (vertices of the simplex).
        Must be at least 2.
    n_samples : int
        The number of samples.

    Returns
    -------
    torch.Tensor
        Array of sample points in shape (n_samples, n_params).

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If ``n_params`` is less than 2 or ``n_samples`` is negative.
    """
    if n_params < 2:
        raise ValueError(f"n_params must be at least 2 for Sobol sampling, got {n_params}")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_samples == 0:
        return torch.empty((0, n_params), dtype=torch.float32)

    import numpy as np
    try:
        from scipy.stats import qmc
    except ImportError as e:
        raise ImportError(
            "SciPy is required for simplex_sobol. "
            "Install it with: pip install scipy"
        ) from e

    # Sobol sequence generator for (n_params - 1) dimensions in [0, 1]
    sampler = qmc.Sobol(d=n_params - 1, scramble=True)
    q = sampler.random(n=n_samples)

    # Transform to simplex: q[i] sorted are the spacings? No, that's not Sobol on simplex.
    # Standard way to project onto simplex: use the "uniform mapping"
    # q_i are in [0, 1]. Sort them in each sample.
    q = np.sort(q, axis=1)
    q = np.concatenate(
        [np.zeros((n_samples, 1)), q, np.ones((n_samples, 1))], axis=1
    )
    diffs = np.diff(q, axis=1)

    return torch.from_numpy(diffs).float()


