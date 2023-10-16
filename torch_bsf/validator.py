import json
import typing

from jsonschema import ValidationError, validate


def validate_skeleton(instance: object, dimension: int, degree: int) -> None:
    r"""Validate an instance that has appropriate dimension and degree.

    Parameters
    ----------
    instance
        An index list of a Bezier simplex.
    dimension
        The dimension of a Bezier simplex.
    degree
        The degree of a Bezier simplex.

    Raises
    ------
    ValidationError
        If ``instance`` does not comply with the following schema
        or has an inner array whose sum is not equal to ``degree``.
        ```
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": degree,
                },
                "minItems": dimension,
                "maxItems": dimension,
            },
        }
        ```
    """
    schema = skeleton_schema(dimension, degree)
    validate(instance, schema)
    indices = typing.cast(typing.List[typing.List[int]], instance)

    if dimension == 0:
        return  # no need to check sum since indices is a list of empty lists

    for index in indices:
        s = sum(index)
        if s != degree:
            raise ValidationError(
                f"sum({index})=={s}, which is not equal to degree {degree}."
            )


def skeleton_schema(dimension: int, degree: int) -> typing.Dict[str, typing.Any]:
    r"""Generate a JSON schema for skeletons of the bezier simplex with given ``dimension`` and ``degree``.

    Parameters
    ----------
    dimension
        The dimension of a Bezier simplex.
    degree
        The degree of a Bezier simplex.

    Returns
    -------
        A JSON schema.

    Raises
    ------
    ValueError
        If ``dimension`` or ``degree`` is negative.

    See Also
    --------
    validate_skeleton
        Validate an instance that has appropriate dimension and degree.
    """
    if dimension < 0:
        raise ValueError(f"dimension must be non-negative, but {dimension} is given.")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, but {degree} is given.")

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": degree,
            },
            "minItems": dimension,
            "maxItems": dimension,
        },
    }


def skeleton(val: str) -> typing.List[typing.List[int]]:
    r"""Parse ``val`` into a skeleton.

    Parameters
    ----------
    val
        A string expression of a skeleton.

    Returns
    -------
        The persed skeleton.
    """
    val = val.replace("(", "[").replace(")", "]").replace("{", "[").replace("}", "]")
    skeleton = typing.cast(typing.List[typing.List[int]], json.loads(val))
    return skeleton


def int_or_str(val: str) -> typing.Union[int, str]:
    r"""Try to parse int.
    Return the int value if the parse is succeeded; the original string otherwise.

    Parameters
    ----------
    val
        The value to try to convert into int.

    Returns
    -------
    typing.Union[int, str]
        The converted integer or the original value.
    """
    try:
        return int(val)
    except ValueError:
        return val
