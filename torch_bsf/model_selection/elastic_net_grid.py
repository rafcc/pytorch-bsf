from argparse import ArgumentParser

import numpy as np


def reverse_logspace(num: int = 50, base: int = 10) -> np.ndarray:
    """Return numbers spaced evenly on a log scale.

    Parameters
    ----------
    num : integer, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    base : array_like, optional
        The base of the log space.
        The step size between the elements in ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    Examples
    --------
    >>> reverse_logspace(0)
    array([], dtype=float64)
    >>> reverse_logspace(1)
    array([0.])
    >>> reverse_logspace(2)
    array([0.        , 0.75974693])
    >>> reverse_logspace(3)
    array([0.        , 0.59537902, 0.87172948])
    >>> reverse_logspace(num=5, base=10)
    array([0.        , 0.4100474 , 0.66876981, 0.83201262, 0.93501187])
    >>> reverse_logspace(num=5, base=100)
    array([0.        , 0.60797255, 0.85001079, 0.94636795, 0.98472842])
    >>> reverse_logspace(num=5, base=1000)
    array([0.        , 0.74956092, 0.93784211, 0.9851362 , 0.99701594])
    """
    return 1.0 - (np.logspace(1.0, 0.0, num, endpoint=False, base=base) - 1.0) / (
        base - 1
    )


def elastic_net_grid(n_lambdas: int = 102, n_alphas: int = 12, n_vertex_copies: int = 1, base: int = 10) -> np.ndarray:
    """Return an array of 3D grid points on the standard 2-simplex, which is suitable for grid search for elastic net's hyperparameters.

    The returned array is of shape ``((n_lambdas - 1) * n_alphas + 3 * n_copy_vertices - 2, 3)``.
    The first column values are spaced evenly on a log scale.
    The second and third column values are spaced evenly over a specified interval.

    Parameters
    ----------
    n_lambdas : int, optional
        Number of samples to generate along `lambda` axis.
        The values are equally spaced on a log scale.
        Default is ``102``. Must be non-negative.
    n_alphas : int, optional
        Number of samples to generate along `alpha` axis.
        The values are equally spaced.
        Default is ``12``. Must be non-negative.
    n_copy_vertices : int, optional
        Number of duplicated samples to generate vertices.
        Each vertices is sampled ``n_copy_vertices`` times.
        Default is ``1``. Must be non-negative.
        Useful for k-fold cross validation.
    base : int, optional
        The base of the log space.
        The step size between the elements in ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.

    Returns
    -------
    samples : ndarray
        ``(n_lambdas - 1) * n_alphas + 3 * n_copy_vertices - 2`` samples.

    Examples
    --------
    >>> elastic_net_grid(0, 0)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(0, 1)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(1, 0)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(1, 1)
    array([[1., 0., 0.]])
    >>> elastic_net_grid(1, 2)
    array([[1., 0., 0.]])
    >>> elastic_net_grid(2, 1)
    array([[0., 0., 1.],
           [1., 0., 0.]])
    >>> elastic_net_grid(2, 2)
    array([[0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    >>> elastic_net_grid(2, 3)
    array([[0. , 0. , 1. ],
           [0. , 0.5, 0.5],
           [0. , 1. , 0. ],
           [1. , 0. , 0. ]])
    >>> elastic_net_grid(3, 3)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ]])
    >>> elastic_net_grid(3, 3, 2)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])
    >>> elastic_net_grid(3, 3, 3)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])
    """
    if n_lambdas < 1 or n_alphas < 1:
        return np.empty((0, 3))
    w1_values = reverse_logspace(n_lambdas - 1, base)
    w2_values = np.linspace(0.0, 1.0 - w1_values, n_alphas, axis=1)
    n_vertices = 3 * n_vertex_copies - 2

    grids = np.zeros(((n_lambdas - 1) * n_alphas + n_vertices, 3))
    grids[:-n_vertices, 0] = np.repeat(w1_values, n_alphas)
    grids[:-n_vertices, 1] = w2_values.reshape(-1)  # faster than ravel and flatten
    grids[:-n_vertices, 2] = 1 - grids[:-n_vertices, 0] - grids[:-n_vertices, 1]
    grids[-n_vertices] = [1.0, 0.0, 0.0]  # last entry

    if n_vertices > 1:
        # fill remaining space with vertex copies
        grids[-n_vertices + 1:] = np.tile(np.eye(3), (n_vertex_copies - 1, 1))

    return grids


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="python -m torch_bsf.model_selection.elastic_net_grid",
        description="Grid search for elastic net's hyperparameters",
    )
    parser.add_argument("--n_lambdas", help="Number of samples for lambda values", type=int, default=102)
    parser.add_argument("--n_alphas", help="Number of samples for alpha values", type=int, default=12)
    parser.add_argument("--n_vertex_copies", help="Number of copies of each vertex", type=int, default=10)
    parser.add_argument("--base", help="Base of the log space", type=int, default=10)
    args = parser.parse_args()

    grid = elastic_net_grid(args.n_lambdas, args.n_alphas, args.n_vertex_copies, base=args.base)
    print("\n".join(f"{p[0]:.17e},{p[1]:.17e},{p[2]:.17e}" for p in grid.tolist()))
