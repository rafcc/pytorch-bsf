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
