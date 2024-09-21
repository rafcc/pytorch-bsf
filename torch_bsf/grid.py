import numpy as np

def reverse_logspace(num=50, base=10):
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
    return 1.0 - (np.logspace(1.0, 0.0, num, endpoint=False, base=base) - 1.0) / (base - 1)


def elastic_net_grid(num=(100, 10), base=10):
    """Return an array of 3D grid points on the standard 2-simplex, which is suitable for grid search for elastic net's hyperparameters.

    The returned array is of shape ``(num[0] * num[1], 3)``.
    The first column values are spaced evenly on a log scale.
    The second and third column values are spaced evenly over a specified interval.

    Examples
    --------
    >>> elastic_net_grid((0, 0))
    array([[1., 0., 0.]])
    >>> elastic_net_grid((0, 1))
    array([[1., 0., 0.]])
    >>> elastic_net_grid((1, 0))
    array([[1., 0., 0.]])
    >>> elastic_net_grid((1, 1))
    array([[0., 0., 1.],
           [1., 0., 0.]])
    >>> elastic_net_grid((1, 1))
    array([[0., 0., 1.],
           [1., 0., 0.]])
    >>> elastic_net_grid((1, 2))
    array([[0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    >>> elastic_net_grid((2, 1))
    array([[0.        , 0.        , 1.        ],
           [0.75974693, 0.        , 0.24025307],
           [1.        , 0.        , 0.        ]])
    >>> elastic_net_grid((2, 2))
    array([[0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ]])
    >>> elastic_net_grid((2, 3))
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ]])
    >>> elastic_net_grid((3 , 3))
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.59537902, 0.        , 0.40462098],
           [0.59537902, 0.20231049, 0.20231049],
           [0.59537902, 0.40462098, 0.        ],
           [0.87172948, 0.        , 0.12827052],
           [0.87172948, 0.06413526, 0.06413526],
           [0.87172948, 0.12827052, 0.        ],
           [1.        , 0.        , 0.        ]])
    """
    w1_values = reverse_logspace(num[0], base)
    w2_values = np.linspace(0, 1 - w1_values, num[1], axis=1)

    grids = np.zeros((num[0] * num[1] + 1, 3))
    grids[:-1, 0] = np.repeat(w1_values, num[1])
    grids[:-1, 1] = w2_values.reshape(-1)  # faster than ravel and flatten
    grids[:-1, 2] = 1 - grids[:-1, 0] - grids[:-1, 1]
    grids[-1] = [1.0, 0.0, 0.0]  # last entry

    return grids


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
        prog="python -m torch_bsf.grid", description="Bezier simplex fitting"
    )
    parser.add_argument("--num1", type=int, default=100)
    parser.add_argument("--num2", type=int, default=10)
    parser.add_argument("--base", type=int, default=10)
    args = parser.parse_args()
    
    grid = elastic_net_grid(num=(args.num1, args.num2), base=args.base)
    print("\n".join(f"{p[0]:.17e},{p[1]:.17e},{p[2]:.17e}" for p in grid.tolist()))