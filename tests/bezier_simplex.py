import pytest
import torch

import torch_bsf as tb
import torch_bsf.bezier_simplex as tbbs


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_zeros(n_params: int, n_values: int, degree: int):
    bs = tbbs.zeros(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_rand(n_params: int, n_values: int, degree: int):
    bs = tbbs.rand(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_randn(n_params: int, n_values: int, degree: int):
    bs = tbbs.randn(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, degree",
    ((n_params, degree) for n_params in range(3) for degree in range(3)),
)
def test_simplex_indices(n_params: int, degree: int):
    indices = list(tbbs.simplex_indices(n_params, degree))
    if n_params <= 1 or degree == 0:
        assert len(indices) == 1
    else:
        assert len(indices) > 1

    if n_params == 0:
        assert indices[0] == ()
        assert indices[-1] == ()
    else:
        assert indices[0] == (degree,) + (0,) * (n_params - 1)
        assert indices[-1] == (0,) * (n_params - 1) + (degree,)


@pytest.mark.parametrize(
    "data",
    (
        {str(index): [0] for index in tbbs.simplex_indices(0, 1)},
        {str(index): [0] for index in tbbs.simplex_indices(1, 2)},
        {str(index): [0] for index in tbbs.simplex_indices(2, 3)},
    ),
)
def test_validate_control_points(data):
    tbbs.validate_control_points(data)


def test_fit():
    ts = torch.tensor(  # parameters on a simplex
        [
            [3 / 3, 0 / 3, 0 / 3],
            [2 / 3, 1 / 3, 0 / 3],
            [2 / 3, 0 / 3, 1 / 3],
            [1 / 3, 2 / 3, 0 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 0 / 3, 2 / 3],
            [0 / 3, 3 / 3, 0 / 3],
            [0 / 3, 2 / 3, 1 / 3],
            [0 / 3, 1 / 3, 2 / 3],
            [0 / 3, 0 / 3, 3 / 3],
        ]
    )
    xs = 1 - ts * ts  # values corresponding to the parameters

    # Train a model
    bs = tb.fit(params=ts, values=xs, degree=3)

    # Predict by the trained model
    t = [[0.2, 0.3, 0.5]]
    x = bs(t)
    print(f"{t} -> {x}")


@pytest.mark.parametrize(
    "init_type",
    ("instance", "rand", "file"),
)
def test_partial_fit(init_type):
    ts = torch.tensor(  # parameters on a simplex
        [
            [8 / 8, 0 / 8],
            [7 / 8, 1 / 8],
            [6 / 8, 2 / 8],
            [5 / 8, 3 / 8],
            [4 / 8, 4 / 8],
            [3 / 8, 5 / 8],
            [2 / 8, 6 / 8],
            [1 / 8, 7 / 8],
            [0 / 8, 8 / 8],
        ]
    )
    xs = 1 - ts * ts  # values corresponding to the parameters

    if init_type == "instance":
        # Initialize 2D control points of a Bezier triangle of degree 3
        init = {
            # index: value
            (3, 0): [0.0, 0.1],
            (2, 1): [1.0, 1.1],
            (1, 2): [2.0, 2.1],
            (0, 3): [3.0, 3.1],
        }
    elif init_type == "rand":
        # Or, generate random control points in [0, 1)
        init = tbbs.rand(n_params=2, n_values=2, degree=3)
    elif init_type == "file":
        # Or, load control points from a file
        init = tbbs.load("control_points.yml")
    else:
        raise ValueError()
    # Train the edge of a Bezier curve while its vertices are fixed
    bs = tbbs.fit(
        params=ts,  # input observations (training data)
        values=xs,  # output observations (training data)
        init=init,  # initial values of control points
        fix=[[3, 0], [0, 3]],  # fix vertices of the Bezier curve
    )

    # Predict by the trained model
    t = [
        [0.2, 0.8],
        [0.7, 0.3],
    ]
    x = bs(t)
    print(f"{t} -> {x}")
