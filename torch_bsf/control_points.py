import typing

import torch
import torch.nn as nn
import typing_extensions

Index: typing_extensions.TypeAlias = typing.Union[
    str, typing.Sequence[int], torch.Tensor
]
r"""The index type of control points of a Bezier simplex."""

Value: typing_extensions.TypeAlias = typing.Union[typing.Sequence[float], torch.Tensor]
r"""The value type of control points of a Bezier simplex."""

ControlPointsData: typing_extensions.TypeAlias = typing.Dict[Index, Value]
r"""The data type of control points of a Bezier simplex."""


def indices(dim: int, deg: int) -> typing.Iterable[typing.Tuple[int]]:
    r"""Iterates the index of control points of the Bezier simplex.

    Parameters
    ----------
    dim
        The array length of indices.
    deg
        The degree of the Bezier simplex.

    Returns
    -------
    The indices.
    """

    def iterate(c, r):
        if len(c) == dim - 1:
            yield c + (r,)
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i,), r - i)

    yield from iterate((), deg)


def to_parameterdict_key(index: Index) -> str:
    """Convert an index to a key of a ParameterDict.

    Args:
        index (str or tuple[int]): An index of a ParameterDict.

    Returns:
        str: A key of a ParameterDict.
    """
    if isinstance(index, str):
        # If index is a string, it is already a key of a ParameterDict.
        return index.replace("(", "[").replace(")", "]").replace(",]", "]")
    if hasattr(index, "tolist"):
        # If index is a tensor or array, convert it to a string.
        return str(index.tolist())
    # If index is a tuple, convert it to a string.
    return str(list(index))


def to_parameterdict_value(value: Value) -> torch.Tensor:
    """Convert a value to a value of a ParameterDict.

    Args:
        value (list[float]): A value of a ParameterDict.

    Returns:
        torch.Tensor: A value of a ParameterDict.
    """
    return torch.as_tensor(value)


def to_parameterdict(data: typing.Dict[Index, Value]) -> typing.Dict[str, torch.Tensor]:
    """Convert data to a dictionary of parameters.

    Args:
        data (dict): Data to be converted.

    Returns:
        dict: Converted data.
    """
    return {
        to_parameterdict_key(index): to_parameterdict_value(value)
        for index, value in data.items()
    }


class ControlPoints(nn.ParameterDict):
    """Control points of a Bezier simplex.

    Args:
        degree (int): Degree of the Bezier simplex.
        dimension (int): Dimension of the Bezier simplex.
        dtype (torch.dtype): Data type of the control points.
        device (torch.device): Device of the control points.
        requires_grad (bool): Whether to enable gradient computation.
        names (list[str]): Names of the control points.
    """

    def __init__(
        self,
        data: ControlPointsData = {},
    ):
        super().__init__(to_parameterdict(data))
        if len(data) == 0:
            self.degree = 0
            self.n_params = 0
            self.n_values = 0
        else:
            index, value = next(iter(data.items()))
            if isinstance(index, str):
                index = typing.cast(typing.Tuple[int], eval(index))
            self.degree = sum(index)
            self.n_params = len(index)
            self.n_values = len(value)

    def __getitem__(self, key: Index) -> nn.Parameter:
        key = to_parameterdict_key(key)
        return typing.cast(nn.Parameter, super().__getitem__(key))

    def __setitem__(self, key: Index, value: Value) -> None:
        key = to_parameterdict_key(key)
        value = to_parameterdict_value(value)
        super().__setitem__(key, value)

    def indices(self) -> typing.Iterable[typing.Tuple[int]]:
        """Iterates the index of control points of the Bezier simplex.

        Returns
        -------
            The indices.
        """
        return indices(self.n_params, self.degree)
