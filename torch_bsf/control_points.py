from ast import literal_eval
from typing import Iterable, Iterator, Sequence, TypeAlias, cast

import torch
import torch.nn as nn

Index: TypeAlias = str | Sequence[int] | torch.Tensor
r"""The index type of control points of a Bezier simplex."""

Value: TypeAlias = Sequence[float] | torch.Tensor
r"""The value type of control points of a Bezier simplex."""

ControlPointsData: TypeAlias = (
    # we can't use dict[Index, Value] because TypeVar of Dict is invariant.
    dict[str, torch.Tensor]
    | dict[str, list[float]]
    | dict[str, tuple[float, ...]]
    | dict[tuple[int, ...], torch.Tensor]
    | dict[tuple[int, ...], list[float]]
    | dict[tuple[int, ...], tuple[float, ...]]
    | dict[torch.Tensor, torch.Tensor]
    | dict[torch.Tensor, list[float]]
    | dict[torch.Tensor, tuple[float, ...]]
)
r"""The data type of control points of a Bezier simplex."""


def simplex_indices(n_params: int, degree: int) -> Iterable[tuple[int, ...]]:
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
        yield cast(tuple[int, ...], ())
        return

    def iterate(c: tuple[int, ...], r: int) -> Iterable[tuple[int, ...]]:
        if len(c) == n_params - 1:
            yield c + (r,)
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i,), r - i)

    yield from iterate((), degree)


def to_parameterdict_key(index: Index) -> str:
    """Convert an index to a canonical string key.

    Args:
        index (str or tuple[int]): An index of control points.

    Returns:
        str: A canonical string key.
    """
    if isinstance(index, str):
        obj = literal_eval(index)
        if isinstance(obj, int):
            return f"({obj},)"  # str(tuple(obj)) will be error because obj is not iterable.
        # remove non-required trailing comma (one in 2 or more sized tuple) and square brackets.
        return str(tuple(obj))
    if hasattr(index, "tolist"):
        # If index is a tensor or array, convert it to a string.
        return str(tuple(index.tolist()))
    # If index is a tuple, convert it to a string.
    return str(tuple(index))


def to_parameterdict_value(value: Value) -> torch.Tensor:
    """Convert a value to a tensor.

    Args:
        value (list[float]): A value.

    Returns:
        torch.Tensor: A tensor.
    """
    return torch.as_tensor(value)


def to_parameterdict(data: ControlPointsData) -> dict[str, torch.Tensor]:
    """Convert data to a dictionary with canonical string keys.

    Args:
        data (dict): Data to be converted.

    Returns:
        dict: Converted data.
    """
    return {
        to_parameterdict_key(index): to_parameterdict_value(value)
        for index, value in data.items()
    }


class ControlPoints(nn.Module):
    """Control points of a Bezier simplex stored as a single parameter matrix.

    All control points are stored in one ``nn.Parameter`` matrix of shape
    ``(n_indices, n_values)``.  This eliminates the Python-level loop and
    ``torch.stack`` call that would otherwise occur on every forward pass,
    giving a direct O(1) access path to the parameter data.

    Attributes
    ----------
    matrix
        ``nn.Parameter`` of shape ``(n_indices, n_values)`` holding all
        control points in canonical simplex-index order.
    degree
        The degree of the Bezier simplex.
    n_params
        The number of parameters (source dimension + 1).
    n_values
        The number of values (target dimension).

    Examples
    --------
    >>> import torch_bsf
    >>> control_points = torch_bsf.control_points.ControlPoints({
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
    tensor([0.0000, 0.1000, 0.2000], grad_fn=<SelectBackward0>)
    >>> control_points[(0, 1)]
    tensor([1.0000, 1.1000, 1.2000], grad_fn=<SelectBackward0>)

    """

    def __init__(self, data: ControlPointsData | None = None):
        """Initialize the control points of a Bezier simplex.

        The structure of control points is inferred from the data.

        Parameters
        ----------
        data
            The data of control points.
        """
        super().__init__()
        data = data or {}

        if len(data) == 0:
            self.degree = 0
            self.n_params = 0
            self.n_values = 0
            self._indices: list[tuple[int, ...]] = [()]
            self._index_to_row: dict[tuple[int, ...], int] = {(): 0}
            self.matrix = nn.Parameter(torch.empty(1, 0))
            return

        first_key, first_val = next(iter(data.items()))
        if isinstance(first_key, str):
            parsed_key = cast(tuple[int, ...], tuple(literal_eval(first_key)))
        elif hasattr(first_key, "tolist"):
            parsed_key = tuple(int(x) for x in first_key.tolist())  # type: ignore[union-attr]
        else:
            parsed_key = tuple(int(x) for x in first_key)

        self.degree = sum(parsed_key)
        self.n_params = len(parsed_key)
        self.n_values = len(first_val)

        self._indices = list(simplex_indices(self.n_params, self.degree))
        self._index_to_row = {idx: row for row, idx in enumerate(self._indices)}

        # Build the parameter matrix in canonical index order.
        # Missing indices (partial data) are filled with zeros.
        normalized = to_parameterdict(data)
        rows = [
            normalized.get(to_parameterdict_key(idx), torch.zeros(self.n_values))
            for idx in self._indices
        ]
        self.matrix = nn.Parameter(
            torch.stack(rows).to(torch.get_default_dtype())
        )

    def extra_repr(self) -> str:
        return f"n_params={self.n_params}, degree={self.degree}, n_values={self.n_values}"

    def _key_to_row(self, key: Index) -> int:
        """Convert an index to its matrix row number."""
        k = to_parameterdict_key(key)
        idx = cast(tuple[int, ...], tuple(literal_eval(k)))
        return self._index_to_row[idx]

    def __getitem__(self, key: Index) -> torch.Tensor:
        return self.matrix[self._key_to_row(key)]

    def __setitem__(self, key: Index, value: Value) -> None:
        row = self._key_to_row(key)
        v = to_parameterdict_value(value).to(dtype=self.matrix.dtype, device=self.matrix.device)
        with torch.no_grad():
            self.matrix.data[row] = v

    def __contains__(self, key: object) -> bool:
        try:
            self._key_to_row(cast(Index, key))
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def __len__(self) -> int:
        return len(self._indices)

    def indices(self) -> Iterator[tuple[int, ...]]:
        """Iterates the index of control points of the Bezier simplex.

        Returns
        -------
            The indices in canonical order.
        """
        return iter(self._indices)

    def keys(self) -> Iterator[str]:
        """Iterates canonical string keys in canonical order."""
        for idx in self._indices:
            yield to_parameterdict_key(idx)

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterates ``(str_key, value_tensor)`` pairs in canonical order.

        Returns
        -------
            An iterator of ``(key, value)`` pairs.
        """
        for idx in self._indices:
            yield to_parameterdict_key(idx), self.matrix[self._index_to_row[idx]]
