import typing

import numpy as np
import pytest
import torch

from torch_bsf.control_points import (
    ControlPoints,
    ControlPointsData,
    simplex_indices,
    to_parameterdict_key,
)


def test_to_parameterdict_key():
    EMPTY = "()"
    assert to_parameterdict_key([]) == EMPTY
    assert to_parameterdict_key(()) == EMPTY
    assert to_parameterdict_key(np.array([])) == EMPTY
    assert to_parameterdict_key(torch.tensor([])) == EMPTY
    assert to_parameterdict_key("[]") == EMPTY
    assert to_parameterdict_key("()") == EMPTY

    ONE_SIZED = "(1,)"
    assert to_parameterdict_key([1]) == ONE_SIZED
    assert to_parameterdict_key((1,)) == ONE_SIZED
    assert to_parameterdict_key(np.array([1])) == ONE_SIZED
    assert to_parameterdict_key(torch.tensor([1])) == ONE_SIZED
    assert to_parameterdict_key("[1,]") == ONE_SIZED
    assert to_parameterdict_key("(1,)") == ONE_SIZED
    assert to_parameterdict_key("[1]") == ONE_SIZED
    assert to_parameterdict_key("(1)") == ONE_SIZED

    TWO_SIZED = "(1, 2)"
    assert to_parameterdict_key([1, 2]) == TWO_SIZED
    assert to_parameterdict_key((1, 2)) == TWO_SIZED
    assert to_parameterdict_key(np.array([1, 2])) == TWO_SIZED
    assert to_parameterdict_key(torch.tensor([1, 2])) == TWO_SIZED
    assert to_parameterdict_key("[1, 2,]") == TWO_SIZED
    assert to_parameterdict_key("(1, 2,)") == TWO_SIZED
    assert to_parameterdict_key("[1, 2]") == TWO_SIZED
    assert to_parameterdict_key("(1, 2)") == TWO_SIZED


@pytest.mark.parametrize(
    "data, degree, n_params, n_values",
    (
        (
            {
                (1, 0): [0.0],
                (0, 1): [1.0],
            },
            1,
            2,
            1,
        ),
        (
            {
                (1, 0): [0.0, 0.1],
                (0, 1): [1.0, 1.1],
            },
            1,
            2,
            2,
        ),
        (
            {
                (2, 0): [0.0],
                (0, 2): [1.0],
                (1, 1): [1.0],
            },
            2,
            2,
            1,
        ),
        (
            {
                (1, 0, 0): [0.0],
                (0, 1, 0): [1.0],
            },
            1,
            3,
            1,
        ),
        (
            {
                (2, 0, 0): [0.0],
                (1, 1, 0): [1.0],
            },
            2,
            3,
            1,
        ),
    ),
)
def test_control_points___init__(
    data: ControlPointsData,
    degree: int,
    n_params: int,
    n_values: int,
) -> None:
    cps = ControlPoints(data=data)
    assert len(cps) == len(list(simplex_indices(n_params, degree)))
    assert cps.degree == degree
    assert cps.n_params == n_params
    assert cps.n_values == n_values


@pytest.mark.parametrize(
    "data, index, value",
    (
        (
            {
                (1, 0): [0.0],
                (0, 1): [1.0],
            },
            (0, 1),
            torch.tensor([1.0]),
        ),
        (
            {
                (1, 0): [0.0],
                (0, 1): [1.0],
            },
            [0, 1],
            torch.tensor([1.0]),
        ),
        (
            {
                (1, 0): [0.0],
                (0, 1): [1.0],
            },
            range(2),
            torch.tensor([1.0]),
        ),
        (
            {
                (1, 0): [0.0],
                (0, 1): [1.0],
            },
            torch.tensor([1, 0]),
            torch.tensor([0.0]),
        ),
        (
            {
                (1, 0, 0): [0.0, 0.1, 0.2],
                (0, 1, 0): [1.0, 1.1, 1.2],
            },
            (0, 1, 0),
            torch.tensor([1.0, 1.1, 1.2]),
        ),
    ),
)
def test_control_points___getitem__(
    data: ControlPointsData,
    index: typing.Tuple[int, ...],
    value: torch.Tensor,
):
    cps = ControlPoints(data=data)
    assert torch.allclose(cps[index].float(), value.float())


def test_control_points_matrix_is_parameter():
    """matrix attribute should be a leaf nn.Parameter, not a computed tensor."""
    import torch.nn as nn

    cps = ControlPoints({(1, 0): [0.0, 1.0], (0, 1): [2.0, 3.0]})
    assert isinstance(cps.matrix, nn.Parameter)
    assert cps.matrix.requires_grad
    # Accessing matrix twice returns the same object (no recomputation).
    assert cps.matrix is cps.matrix


def test_simplex_indices_negative_n_params():
    from torch_bsf.control_points import simplex_indices

    with pytest.raises(ValueError, match="n_params must be non-negative"):
        list(simplex_indices(-1, 1))


def test_simplex_indices_negative_degree():
    from torch_bsf.control_points import simplex_indices

    with pytest.raises(ValueError, match="degree must be non-negative"):
        list(simplex_indices(1, -1))


def test_control_points_empty_init():
    """ControlPoints with no data should have degree=0, n_params=0, n_values=0."""
    import torch.nn as nn

    cps = ControlPoints({})
    assert cps.degree == 0
    assert cps.n_params == 0
    assert cps.n_values == 0
    assert len(cps) == 1
    assert isinstance(cps.matrix, nn.Parameter)
    assert cps.matrix.shape == (1, 0)


def test_control_points_tensor_key():
    """ControlPoints should accept tensor keys (hasattr .tolist path)."""
    cps = ControlPoints({
        torch.tensor([1, 0]): [0.0, 1.0],
        torch.tensor([0, 1]): [2.0, 3.0],
    })
    assert cps.degree == 1
    assert cps.n_params == 2


def test_control_points_2d_value_raises():
    """ControlPoints should raise ValueError if a value is a 2-D tensor."""
    with pytest.raises(ValueError, match="1-D"):
        ControlPoints({(1, 0): torch.ones(2, 2)})


def test_control_points_extra_repr():
    """extra_repr should return a formatted string with dimensions."""
    cps = ControlPoints({(1, 0): [0.0], (0, 1): [1.0]})
    r = repr(cps)
    assert "n_params=2" in r
    assert "degree=1" in r
    assert "n_values=1" in r


def test_control_points_setitem():
    """__setitem__ should update the underlying matrix in-place."""
    cps = ControlPoints({(1, 0): [0.0, 0.0], (0, 1): [1.0, 1.0]})
    cps[(1, 0)] = [5.0, 6.0]
    assert torch.allclose(cps[(1, 0)].float(), torch.tensor([5.0, 6.0]))


def test_control_points_contains_valid():
    """__contains__ should return True for valid keys."""
    cps = ControlPoints({(1, 0): [0.0], (0, 1): [1.0]})
    assert (1, 0) in cps
    assert (0, 1) in cps


def test_control_points_contains_invalid():
    """__contains__ should return False for keys not in the control points."""
    cps = ControlPoints({(1, 0): [0.0], (0, 1): [1.0]})
    assert (2, 0) not in cps
    assert "invalid" not in cps

