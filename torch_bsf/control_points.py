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

ControlPointsData: typing_extensions.TypeAlias = typing.Union[
    # we can't use typing.Dict[Index, Value] because TypeVar of Dict is invariant.
    typing.Dict[str, torch.Tensor],
    typing.Dict[str, typing.List[float]],
    typing.Dict[str, typing.Tuple[float, ...]],
    typing.Dict[typing.Tuple[int, ...], torch.Tensor],
    typing.Dict[typing.Tuple[int, ...], typing.List[float]],
    typing.Dict[typing.Tuple[int, ...], typing.Tuple[float, ...]],
    typing.Dict[torch.Tensor, torch.Tensor],
    typing.Dict[torch.Tensor, typing.List[float]],
    typing.Dict[torch.Tensor, typing.Tuple[float, ...]],
]
r"""The data type of control points of a Bezier simplex."""


def simplex_indices(
    n_params: int, degree: int
) -> typing.Iterable[typing.Tuple[int, ...]]:
    r"""Iterates the index of control points of a Bezier simplex.

    Parameters
    ----------
    n_params
        The tuple length of each index.
    degree
        The degree of the Bezier simplex.

    Returns
    -------
    The indices.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, but {n_params} is given.")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, but {degree} is given.")
    if n_params == 0:
        yield typing.cast(typing.Tuple[int, ...], ())
        return

    def iterate(
        c: typing.Tuple[int, ...],
        r: int,
    ) -> typing.Iterable[typing.Tuple[int, ...]]:
        if len(c) == n_params - 1:
            yield c + (r,)
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i,), r - i)

    yield from iterate((), degree)


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


def to_parameterdict(data: ControlPointsData) -> typing.Dict[str, torch.Tensor]:
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

    Attributes
    ----------
    degree
        The degree of the Bezier simplex.
    n_params
        The number of parameters.
    n_values
        The number of values.

    Examples
    --------
    >>> import torch_bsf
    >>> control_points = torch_bsf.ControlPoints({
    ...     (1, 0): [0.0, 0.1, 0.2],
    ...     (0, 1): [1.0, 1.1, 1.2],
    ... })

    >>> control_points.degree
    1
    >>> control_points.n_params
    2
    >>> control_points.n_values
    3

    >>> control_points[(1, 0)]
    Parameter containing:
    tensor([0.0000, 0.1000, 0.2000], requires_grad=True)
    >>> control_points[(0, 1)]
    Parameter containing:
    tensor([1.0000, 1.1000, 1.2000], requires_grad=True)

    >>> control_points[(1, 0)].requires_grad = False

    """

    def __init__(
        self,
        data: ControlPointsData,
    ):
        """Initialize the control points of a Bezier simplex.

        The structure of control points is inferred from the data.

        Parameters
        ----------
        data
            The data of control points.
        """
        super().__init__(to_parameterdict(data))
        if len(data) == 0:
            self.degree = 0
            self.n_params = 0
            self.n_values = 0
        else:
            index, value = typing.cast(
                typing.Tuple[typing.Sequence[int], typing.Sequence[float]],
                next(iter(data.items())),
            )
            if isinstance(index, str):
                index = typing.cast(typing.List[int], eval(index))
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

    def indices(self) -> typing.Iterable[typing.Tuple[int, ...]]:
        """Iterates the index of control points of the Bezier simplex.

        Returns
        -------
            The indices.
        """
        return simplex_indices(self.n_params, self.degree)
