import pytest
from jsonschema import ValidationError

from torch_bsf.validator import (
    index_list,
    indices_schema,
    int_or_str,
    validate_simplex_indices,
)


@pytest.mark.parametrize(
    "val, expected",
    (
        ("-1", -1),
        ("1", 1),
        ("+1", 1),
        ("0", 0),
        ("-0", 0),
        ("+0", 0),
        ("1_0", 10),
        ("0_0", 0),
        ("auto", "auto"),
        ("None", "None"),
        ("null", "null"),
        ("-inf", "-inf"),
        ("inf", "inf"),
        ("+inf", "+inf"),
        ("0.0", "0.0"),
        ("1E2", "1E2"),
        ("1e2", "1e2"),
        ("0b1", "0b1"),
        ("0o1", "0o1"),
        ("0x1", "0x1"),
        ("[]", "[]"),
        ("()", "()"),
        ("{}", "{}"),
        ("", ""),
    ),
)
def test_int_or_str(val, expected):
    assert int_or_str(val) == expected


@pytest.mark.parametrize(
    "val, expected",
    (
        ("[1, 0]", [1, 0]),
        ("(1, 0)", [1, 0]),
        ("{1, 0}", [1, 0]),
        ("()", []),
        ("(())", [[]]),
        ("[()]", [[]]),
    ),
)
def test_index_list(val, expected):
    assert index_list(val) == expected


@pytest.mark.parametrize(
    "n_params, degree",
    (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ),
)
def test_indices_schema(n_params, degree):
    indices_schema(n_params, degree)


@pytest.mark.parametrize(
    "n_params, degree",
    (
        (-1, -1),
        (-1, 0),
        (0, -1),
    ),
)
def test_indices_schema_value_error(n_params, degree):
    with pytest.raises(ValueError):
        indices_schema(n_params, degree)


@pytest.mark.parametrize(
    "n_params, degree, instance",
    (
        (0, 0, []),
        (0, 0, [[]]),
        (0, 0, [[], []]),
        (0, 1, []),
        (0, 1, [[]]),
        (0, 1, [[], []]),
        (1, 0, []),
        (1, 0, [[0]]),
        (1, 0, [[0], [0]]),
        (1, 1, []),
        (1, 1, [[1]]),
        (2, 0, []),
        (2, 0, [[0, 0]]),
        (2, 0, [[0, 0], [0, 0]]),
        (2, 1, []),
        (2, 1, [[1, 0], [0, 1]]),
        (2, 2, []),
        (2, 2, [[2, 0], [1, 1], [0, 2]]),
        (3, 0, []),
        (3, 0, [[0, 0, 0]]),
        (3, 0, [[0, 0, 0], [0, 0, 0]]),
        (3, 1, []),
        (3, 1, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ),
)
def test_validate_simplex_indices(n_params, degree, instance):
    validate_simplex_indices(instance, n_params, degree)


@pytest.mark.parametrize(
    "n_params, degree",
    (
        (-1, -2),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (-1, 2),
    ),
)
def test_validate_simplex_indices_value_error(n_params, degree):
    with pytest.raises(ValueError):
        validate_simplex_indices(None, n_params, degree)


@pytest.mark.parametrize(
    "n_params, degree, instance",
    (
        (0, 0, True),
        (0, 0, False),
        (0, 0, None),
        (0, 0, ""),
        (0, 0, 0),
        (0, 0, [True]),
        (0, 0, [False]),
        (0, 0, [None]),
        (0, 0, [""]),
        (0, 0, [0]),
        (0, 0, [[True]]),
        (0, 0, [[False]]),
        (0, 0, [[None]]),
        (0, 0, [[""]]),
        (0, 0, [[0]]),
        (0, 0, [[0, 0]]),
        (1, 0, [[]]),
        (1, 0, [[0, 0]]),
        (1, 0, [[-1]]),
        (1, 0, [[1]]),
        (1, 1, [[-1]]),
        (1, 1, [[2]]),
        (2, 0, [[0]]),
        (2, 0, [[0, 0, 0]]),
        (2, 0, [[0, -1]]),
        (2, 0, [[0, 1]]),
        (2, 1, [[0, -1]]),
        (2, 1, [[0, 2]]),
        (2, 1, [[0, 0]]),
        (2, 1, [[1, 1]]),
    ),
)
def test_validate_simplex_indices_validation_error(n_params, degree, instance):
    with pytest.raises(ValidationError):
        validate_simplex_indices(instance, n_params, degree)
