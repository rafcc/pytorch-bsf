import typing

import numpy as np
import pytest
import torch

from torch_bsf.control_points import (
    ControlPoints,
    ControlPointsData,
    to_parameterdict_key,
)


def test_to_parameterdict_key():
    assert to_parameterdict_key([]) == "[]"
    assert to_parameterdict_key(()) == "[]"
    assert to_parameterdict_key(np.array([])) == "[]"
    assert to_parameterdict_key(torch.tensor([])) == "[]"
    assert to_parameterdict_key("[]") == "[]"
    assert to_parameterdict_key("()") == "[]"
    assert to_parameterdict_key([1]) == "[1]"
    assert to_parameterdict_key((1,)) == "[1]"
    assert to_parameterdict_key(np.array([1])) == "[1]"
    assert to_parameterdict_key(torch.tensor([1])) == "[1]"
    assert to_parameterdict_key("[1,]") == "[1]"
    assert to_parameterdict_key("(1,)") == "[1]"
    assert to_parameterdict_key([1, 2]) == "[1, 2]"
    assert to_parameterdict_key((1, 2)) == "[1, 2]"
    assert to_parameterdict_key("(1, 2)") == "[1, 2]"
    assert to_parameterdict_key(np.array([1, 2])) == "[1, 2]"
    assert to_parameterdict_key(torch.tensor([1, 2])) == "[1, 2]"


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
    assert len(cps) == len(data)
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
    assert torch.equal(cps[index], value)
