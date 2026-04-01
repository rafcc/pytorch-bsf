import itertools
import warnings

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
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")
    if n_params == 0:
        # Constant simplex: exactly one valid parameter (the empty tuple).
        return torch.empty((1, 0), dtype=torch.float32)
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

    Uses a scrambled Sobol sequence projected onto the simplex via the
    sorted-differences mapping.  Sobol sequences are *low-discrepancy*:
    they fill space more uniformly than pseudo-random draws, giving a
    convergence rate of roughly O((log N)^(d-1) / N) instead of the
    O(1/sqrt(N)) rate of Monte Carlo sampling (where d = ``n_params - 1``).

    .. note::
        **Power-of-two sample sizes are strongly recommended.**
        Sobol sequences are constructed in base 2 and achieve their best
        uniformity guarantees when ``n_samples`` is an exact power of 2
        (e.g. 64, 128, 256, …).  When ``n_samples`` is not a power of 2,
        the strongest low-discrepancy guarantees no longer apply and the
        coverage of the simplex can be somewhat less uniform.  A
        ``UserWarning`` is emitted automatically when a non-power-of-two
        value is requested.

    .. note::
        **scipy is required.**
        This function relies on :class:`scipy.stats.qmc.Sobol`.
        Install it with ``pip install scipy`` or
        ``pip install pytorch-bsf[sampling]``.

    Parameters
    ----------
    n_params : int
        The number of parameters (vertices of the simplex).
        Must be at least 2.  The Sobol sequence is drawn in
        ``n_params - 1`` dimensions and then mapped to the simplex.
    n_samples : int
        The number of samples.  For best coverage, use a power of 2
        (e.g. 64, 128, 256).

    Returns
    -------
    torch.Tensor
        Array of sample points in shape (n_samples, n_params).
        Each row is non-negative and sums to 1.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If ``n_params`` is less than 2 or ``n_samples`` is negative.

    Warns
    -----
    UserWarning
        If ``n_samples`` is not a power of 2.  The samples are still
        returned, but the low-discrepancy coverage guarantee is weakened.

    Examples
    --------
    >>> from torch_bsf.sampling import simplex_sobol
    >>> pts = simplex_sobol(n_params=3, n_samples=128)
    >>> pts.shape
    torch.Size([128, 3])
    >>> pts.sum(dim=1).allclose(torch.ones(128))
    True
    """
    if n_params < 2:
        raise ValueError(f"n_params must be at least 2 for Sobol sampling, got {n_params}")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_samples == 0:
        return torch.empty((0, n_params), dtype=torch.float32)

    # Warn when n_samples is not a power of 2 (Sobol sequences are base-2).
    # Powers of 2 have exactly one bit set, so (n & (n-1)) == 0 iff n is a power of 2.
    if n_samples & (n_samples - 1) != 0:
        warnings.warn(
            f"simplex_sobol: n_samples={n_samples} is not a power of 2. "
            "Sobol sequences achieve their best low-discrepancy coverage "
            "when n_samples is a power of 2 (e.g. 64, 128, 256). "
            "Consider rounding up to the next power of 2 for better uniformity.",
            UserWarning,
            stacklevel=2,
        )

    import numpy as np
    try:
        from scipy.stats import qmc
    except ImportError as e:
        raise ImportError(
            "SciPy is required for simplex_sobol. "
            "Install it with: pip install scipy or pip install pytorch-bsf[sampling]"
        ) from e

    # Sobol sequence generator for (n_params - 1) dimensions in [0, 1].
    # Suppress scipy's own power-of-2 warning; we already emitted a clearer one above.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Sobol'? points require n to be a power of 2",
            category=UserWarning,
        )
        sampler = qmc.Sobol(d=n_params - 1, scramble=True)
        q = sampler.random(n=n_samples)

    # Project Sobol samples to the simplex via the uniform sorted-differences mapping:
    # 1. Sort each (n_params - 1)-dim sample in ascending order.
    # 2. Prepend 0 and append 1 to get n_params + 1 boundary values.
    # 3. Take consecutive differences to obtain n_params non-negative values that sum to 1.
    q = np.sort(q, axis=1)
    q = np.concatenate(
        [np.zeros((n_samples, 1)), q, np.ones((n_samples, 1))], axis=1
    )
    diffs = np.diff(q, axis=1)

    return torch.from_numpy(diffs).float()


