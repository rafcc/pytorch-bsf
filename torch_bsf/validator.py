import json
from typing import cast, Any

from jsonschema import ValidationError, validate


def validate_simplex_indices(instance: object, n_params: int, degree: int) -> None:
    r"""Validate an instance that has appropriate n_params and degree.

    Parameters
    ----------
    instance
        An index list of a Bezier simplex.
    n_params
        The n_params of a Bezier simplex.
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
                "minItems": n_params,
                "maxItems": n_params,
            },
        }
        ```
    """
    schema = indices_schema(n_params, degree)
    validate(instance, schema)
    indices = cast(list[list[int]], instance)

    if n_params == 0:
        return  # no need to check sum since indices is a list of empty lists

    for index in indices:
        s = sum(index)
        if s != degree:
            raise ValidationError(
                f"sum({index})=={s}, which is not equal to degree {degree}."
            )


def indices_schema(n_params: int, degree: int) -> dict[str, Any]:
    r"""Generate a JSON schema for indices of the control points with given ``n_params`` and ``degree``.

    Parameters
    ----------
    n_params
        The number of index elements of control points.
    degree
        The degree of a Bezier surface.

    Returns
    -------
        A JSON schema.

    Raises
    ------
    ValueError
        If ``n_params`` or ``degree`` is negative.

    See Also
    --------
    validate_simplex_indices : Validate an instance that has appropriate n_params and degree.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, but {n_params} is given.")
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
            "minItems": n_params,
            "maxItems": n_params,
        },
    }


def index_list(val: str) -> list[list[int]]:
    r"""Parse ``val`` into a list of indices.

    Parameters
    ----------
    val
        A string expression of a list of indices.

    Returns
    -------
        The persed indices.
    """
    val = val.replace("(", "[").replace(")", "]").replace("{", "[").replace("}", "]")
    indices = cast(list[list[int]], json.loads(val))
    return indices


def int_or_str(val: str) -> int | str:
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
