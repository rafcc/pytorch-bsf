import json
import typing

from jsonschema import validate

from torch_bsf.bezier_simplex import Index


def skeleton_schema(dimension: int, degree: int) -> typing.Dict[str, typing.Any]:
    r"""JSON schema for skeleton of bezier simplex."""
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
    r"""Parse skeleton.

    Parameter
    ---------
    val
        The value to try to convert into int.

    Return
    ------
    typing.List[typing.List[int]]
        The converted integer or the original value.
    """
    val = val.replace("(", "[").replace(")", "]").replace("{", "[").replace("}", "]")
    val = json.loads(val)
    return val


def int_or_str(val: str) -> typing.Union[int, str]:
    r"""Try to parse int.
    Return the int value if the parse is succeeded; the original string otherwise.

    Parameter
    ---------
    val
        The value to try to convert into int.

    Return
    ------
    typing.Union[int, str]
        The converted integer or the original value.
    """
    try:
        return int(val)
    except ValueError:
        return val
