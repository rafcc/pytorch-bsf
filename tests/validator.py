import pytest
from jsonschema import ValidationError

from torch_bsf.validator import int_or_str, skeleton, skeleton_schema, validate_skeleton


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
def test_skeleton(val, expected):
    assert skeleton(val) == expected


@pytest.mark.parametrize(
    "dimension, degree",
    (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ),
)
def test_skeleton_schema(dimension, degree):
    skeleton_schema(dimension, degree)


@pytest.mark.parametrize(
    "dimension, degree",
    (
        (-1, -1),
        (-1, 0),
        (0, -1),
    ),
)
def test_skeleton_schema_value_error(dimension, degree):
    with pytest.raises(ValueError):
        skeleton_schema(dimension, degree)


@pytest.mark.parametrize(
    "dimension, degree, val",
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
def test_validate_skeleton(dimension, degree, val):
    validate_skeleton(val, dimension, degree)


@pytest.mark.parametrize(
    "dimension, degree",
    (
        (-1, -2),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (-1, 2),
    ),
)
def test_validate_skeleton_value_error(dimension, degree):
    with pytest.raises(ValueError):
        validate_skeleton(None, dimension, degree)


@pytest.mark.parametrize(
    "dimension, degree, val",
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
def test_validate_skeleton_validation_error(dimension, degree, val):
    with pytest.raises(ValidationError):
        validate_skeleton(val, dimension, degree)
