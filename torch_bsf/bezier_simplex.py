import csv
import json
from pathlib import Path
import typing
from functools import lru_cache
from math import factorial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim
import yaml
from jsonschema import validate, ValidationError
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch_bsf.control_points import (
    ControlPoints,
    ControlPointsData,
    Index,
    simplex_indices,
    to_parameterdict_key,
)
from torch_bsf.validator import validate_simplex_indices


class BezierSimplexDataModule(pl.LightningDataModule):
    r"""A data module for training a Bezier simplex.

    Parameters
    ----------
    data
        The path to a data file.
    label
        The path to a label file.
    header
        The number of headers in data files.
    batch_size
        The size of minibatch.
    split_ratio
        The ratio of train-val split.
    normalize
        The data normalization method.
        Either ``"max"``, ``"std"``, ``"quantile"``, or ``"none"``.
    """

    def __init__(
        self,
        params: Path,
        values: Path,
        header: int = 0,
        batch_size: typing.Optional[int] = None,
        split_ratio: float = 0.5,
        normalize: str = "none",  # "max", "std", "quantile" or "none"
    ):
        # REQUIRED
        super().__init__()
        self.params = params
        self.values = values
        self.header = header
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.normalize = normalize
        with open(self.params) as f:
            delimiter = "," if self.params.suffix == ".csv" else None
            self.n_params = len(f.readline().split(delimiter))
        with open(self.values) as f:
            delimiter = "," if self.values.suffix == ".csv" else None
            self.n_values = len(f.readline().split(delimiter))

    def setup(self, stage: typing.Optional[str] = None):
        # OPTIONAL
        delimiter = "," if self.params.suffix == ".csv" else None
        params = torch.from_numpy(
            np.loadtxt(self.params, delimiter=delimiter, skiprows=self.header)
        )
        delimiter = "," if self.values.suffix == ".csv" else None
        values = torch.from_numpy(
            np.loadtxt(self.values, delimiter=delimiter, skiprows=self.header)
        )
        if self.normalize == "max":
            mins = values.amin(dim=0)
            maxs = values.amax(dim=0)
            mins[mins == maxs] -= 0.5  # Avoid division by zero
            maxs[mins == maxs] += 0.5  # Avoid division by zero
            values = (values - mins) / (maxs - mins)
        elif self.normalize == "std":
            stds, means = torch.std_mean(values, dim=0)
            stds[stds == 0.0] = 1.0  # Avoid division by zero
            values = (values - means) / stds
        elif self.normalize == "quantile":
            q = 0.05  # Ignore 5% outliers
            mins = values.quantile(q, dim=0)
            maxs = values.quantile(1 - q, dim=0)
            mins[mins == maxs] -= 0.5  # Avoid division by zero
            maxs[mins == maxs] += 0.5  # Avoid division by zero
            values = (values - mins) / (maxs - mins)
        xy = TensorDataset(params, values)
        size = len(xy)
        n_train = int(size * self.split_ratio)
        self.trainset, self.valset = random_split(xy, [n_train, size - n_train])

    def train_dataloader(self) -> DataLoader:
        # REQUIRED
        train_loader = DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size or len(self.trainset),
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        # OPTIONAL
        val_loader = DataLoader(
            self.valset,
            batch_size=self.batch_size or len(self.valset),
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        # OPTIONAL
        return self.val_dataloader()


@lru_cache(1024)
def polynom(degree: int, index: typing.Iterable[int]) -> float:
    r"""Computes a polynomial coefficient :math:`\binom{D}{\mathbf d} = \frac{D!}{d_1!d_2!\cdots d_M!}`.

    Parameters
    ----------
    degree
        The degree :math:`D`.
    index
        The index :math:`\mathbf d`.

    Returns
    -------
    The polynomial coefficient :math:`\binom{D}{\mathbf d}`.
    """
    r: float = factorial(degree)
    for i in index:
        r /= factorial(i)
    return r


def monomial(
    variable: typing.Iterable[float], degree: typing.Iterable[int]
) -> torch.Tensor:
    r"""Computes a monomial :math:`\mathbf t^{\mathbf d} = t_1^{d_1} t_2^{d_2}\cdots t_M^{d^M}`.

    Parameters
    ----------
    variable
        The bases :math:`\mathbf t`.
    degree
        The powers :math:`\mathbf d`.

    Returns
    -------
    The monomial :math:`\mathbf t^{\mathbf d}`.
    """
    v = torch.as_tensor(variable)
    d = torch.as_tensor(degree, device=v.device)
    ret: torch.Tensor = (v**d).prod(dim=-1)
    return ret


class BezierSimplex(pl.LightningModule):
    r"""A Bezier simplex model.

    Parameters
    ----------
    control_points
        The control points of the Bezier simplex.

    Examples
    --------
    >>> ts = torch.tensor(  # parameters on a simplex
    ...     [
    ...         [3/3, 0/3, 0/3],
    ...         [2/3, 1/3, 0/3],
    ...         [2/3, 0/3, 1/3],
    ...         [1/3, 2/3, 0/3],
    ...         [1/3, 1/3, 1/3],
    ...         [1/3, 0/3, 2/3],
    ...         [0/3, 3/3, 0/3],
    ...         [0/3, 2/3, 1/3],
    ...         [0/3, 1/3, 2/3],
    ...         [0/3, 0/3, 3/3],
    ...     ]
    ... )
    >>> xs = 1 - ts * ts  # values corresponding to the parameters
    >>> dl = DataLoader(TensorDataset(ts, xs))
    >>> bs = randn(
    ...     n_params=int(ts.shape[1]),
    ...     n_values=int(xs.shape[1]),
    ...     degree=3,
    ... )
    >>> trainer = pl.Trainer(
    ...     callbacks=[EarlyStopping(monitor="train_mse")],
    ...     enable_progress_bar=False,
    ... )
    >>> trainer.fit(bs, dl)
    >>> ts, xs = bs.meshgrid()

    """

    def __init__(
        self,
        control_points: typing.Union[ControlPoints, ControlPointsData],
    ):
        # REQUIRED
        super().__init__()
        self.control_points = (
            control_points
            if isinstance(control_points, ControlPoints)
            else ControlPoints(control_points)
        )

    @property
    def n_params(self) -> int:
        r"""The number of parameters, i.e., the source dimension + 1."""
        return self.control_points.n_params

    @property
    def n_values(self) -> int:
        r"""The number of values, i.e., the target dimension."""
        return self.control_points.n_values

    @property
    def degree(self) -> int:
        r"""The degree of the Bezier simplex."""
        return self.control_points.degree

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        r"""Process a forwarding step of training.

        Parameters
        ----------
        t
            A minibatch of parameter vectors :math:`\mathbf t`.

        Returns
        -------
        A minibatch of value vectors.
        """
        # REQUIRED
        x = typing.cast(torch.Tensor, 0)  # x will be of Tensor by subsequent broadcast
        for i in simplex_indices(self.n_params, self.degree):
            x += polynom(self.degree, i) * torch.outer(
                monomial(t, i), self.control_points[i]
            )
        return x

    def training_step(self, batch, batch_idx) -> typing.Dict[str, typing.Any]:
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        self.log("train_mse", loss, sync_dist=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx) -> typing.Dict[str, typing.Any]:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("val_mse", mse, sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        return {"val_loss": mse}

    def validation_end(self, outputs) -> typing.Dict[str, typing.Any]:
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.log("val_avg_mse", avg_loss, sync_dist=True)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx) -> typing.Dict[str, typing.Any]:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", mae, sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # REQUIRED
        optimizer = torch.optim.LBFGS(self.parameters())
        return optimizer

    def meshgrid(self, num: int = 100) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a meshgrid of the Bezier simplex.

        Parameters
        ----------
        num
            The number of grid points on each edge.

        Returns
        -------
        ts
            A parameter matrix of the mesh grid.
        xs
            A value matrix of the mesh grid.
        """
        ts = (
            torch.Tensor(list(simplex_indices(n_params=self.n_params, degree=num)))
            / num
        )
        xs = self.forward(ts)
        return ts, xs


def zeros(n_params: int, n_values: int, degree: int) -> BezierSimplex:
    r"""Generates a Bezier simplex with control points at origin.

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.

    Returns
    -------
        A Bezier simplex filled with zeros.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.zeros(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(
          ([0, 2]): Parameter containing: [torch.FloatTensor of size 3]
          ([1, 1]): Parameter containing: [torch.FloatTensor of size 3]
          ([2, 0]): Parameter containing: [torch.FloatTensor of size 3]
      )
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))
    tensor([[0., 0., 0.]], grad_fn=<AddBackward0>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.zeros(n_values) for i in simplex_indices(n_params, degree)}
    )


def rand(n_params: int, n_values: int, degree: int) -> BezierSimplex:
    r"""Generates a random Bezier simplex.

    The control points are initialized by random values.
    The values are uniformly distributed in [0, 1).

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.

    Returns
    -------
        A random Bezier simplex.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.rand(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(
          ([0, 2]): Parameter containing: [torch.FloatTensor of size 3]
          ([1, 1]): Parameter containing: [torch.FloatTensor of size 3]
          ([2, 0]): Parameter containing: [torch.FloatTensor of size 3]
      )
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))  # doctest: +ELLIPSIS
    tensor([[..., ..., ...]], grad_fn=<AddBackward0>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.rand(n_values) for i in simplex_indices(n_params, degree)}
    )


def randn(n_params: int, n_values: int, degree: int) -> BezierSimplex:
    r"""Generates a random Bezier simplex.

    The control points are initialized by random values.
    The values are normally distributed with mean 0 and standard deviation 1.

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.

    Returns
    -------
        A random Bezier simplex.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.randn(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(
          ([0, 2]): Parameter containing: [torch.FloatTensor of size 3]
          ([1, 1]): Parameter containing: [torch.FloatTensor of size 3]
          ([2, 0]): Parameter containing: [torch.FloatTensor of size 3]
      )
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))  # doctest: +ELLIPSIS
    tensor([[..., ..., ...]], grad_fn=<AddBackward0>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.randn(n_values) for i in simplex_indices(n_params, degree)}
    )


def save(path: typing.Union[str, Path], data: BezierSimplex) -> None:
    r"""Saves a Bezier simplex to a file.

    Parameters
    ----------
    path
        The file path to save.
    data
        The Bezier simplex to save.

    Raises
    ------
    ValueError
        If the file type is unknown.

    Examples
    --------
    >>> import torch_bsf
    >>> bs = torch_bsf.randn(n_params=2, n_values=3, degree=2)
    >>> torch_bsf.save("tests/data/bezier_simplex.pt", bs)
    >>> torch_bsf.save("tests/data/bezier_simplex.csv", bs)
    >>> torch_bsf.save("tests/data/bezier_simplex.tsv", bs)
    >>> torch_bsf.save("tests/data/bezier_simplex.json", bs)
    >>> torch_bsf.save("tests/data/bezier_simplex.yml", bs)

    """
    path = Path(path)
    if path.suffix == ".pt":
        torch.save(data, path)

    elif path.suffix == ".csv":
        with open(path, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            for index, value in data.control_points.items():
                writer.writerow([index] + value.tolist())

    elif path.suffix == ".tsv":
        with open(path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for index, value in data.control_points.items():
                writer.writerow([index] + value.tolist())

    elif path.suffix == ".json":
        json.dump(data.control_points, open(path, "w", encoding="utf-8"))

    elif path.suffix in (".yml", ".yaml"):
        yaml.dump(data.control_points, open(path, "w", encoding="utf-8"))

    else:
        raise ValueError(f"Unknown file type: {path}")


CONTROLPOINTS_JSONSCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "patternProperties": {
        r"^\((\d+, *)?\d*\)$": {  # e.g., "(0, 0, 0)": [0.0, 0.0, 0.0]
            "type": "array",
            "items": {
                "type": "number",
            },
        }
    },
    "additionalProperties": False,
}


def validate_control_points(data: typing.Dict[str, typing.List[float]]):
    r"""Validates control points.

    Parameters
    ----------
    data
        The control points.

    Raises
    ------
    ValidationError
        If the control points are invalid.

    Examples
    --------
    >>> from torch_bsf.bezier_simplex import validate_control_points
    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0],
    ... })

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "0, 0, 1": [0.0, 0.0, 1.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: '0, 0, 1' is not valid under any of the given schemas

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0],
    ...     "(0, 0)": [0.0, 0.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: (0, 0)

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1, 0)": [0.0, 0.0, 1.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: (0, 0, 1, 0)

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: [0.0, 0.0, 1.0, 0.0]

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: [0.0, 0.0]
    """
    validate(instance=data, schema=CONTROLPOINTS_JSONSCHEMA)
    index, value = next(iter(data.items()))
    n_params = len(eval(index))
    n_values = len(value)
    for index, value in data.items():
        if len(eval(index)) != n_params:
            raise ValidationError(f"Dimension mismatch: {index}")
        if len(value) != n_values:
            raise ValidationError(f"Dimension mismatch: {value}")


def load(path: typing.Union[str, Path]) -> BezierSimplex:
    r"""Loads a Bezier simplex from a file.

    Parameters
    ----------
    path
        The path to a file.

    Returns
    -------
    A Bezier simplex.

    Raises
    ------
    ValueError
        If the file type is unknown.
    ValidationError
        If the control points are invalid.

    Examples
    --------
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.load("tests/data/bezier_simplex.csv")
    >>> print(bs)
    BezierSimplex(
        (control_points): ControlPoints(
            ([0, 2]): Parameter containing: [torch.FloatTensor of size 3]
            ([1, 1]): Parameter containing: [torch.FloatTensor of size 3]
            ([2, 0]): Parameter containing: [torch.FloatTensor of size 3]
        )
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))
    tensor([[0.0000, 0.0000, 0.0000]], grad_fn=<AddBackward0>)
    """
    cpdata: typing.Dict[str, typing.List[float]]
    path = Path(path)
    if path.suffix == ".pt":
        data = torch.load(path)
        if isinstance(data, BezierSimplex):
            return data
        raise ValueError(f"Unknown data type: {type(data)}")

    elif path.suffix == ".csv":
        cpdata = {
            row[0]: [float(v) for v in row[1:]]
            for row in csv.reader(open(path, encoding="utf-8"))
        }
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif path.suffix == ".tsv":
        cpdata = {
            row[0]: [float(v) for v in row[1:]]
            for row in csv.reader(open(path, encoding="utf-8"), delimiter="\t")
        }
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif path.suffix == ".json":
        cpdata = json.load(open(path, encoding="utf-8"))
        for index, value in cpdata.items():
            cpdata[index] = [float(v) for v in value]
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif path.suffix in (".yml", ".yaml"):
        cpdata = yaml.safe_load(open(path, encoding="utf-8"))
        for index, value in cpdata.items():
            cpdata[index] = [float(v) for v in value]
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    else:
        raise ValueError(f"Unknown file type: {path}")


def fit(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: typing.Optional[int] = None,
    init: typing.Optional[
        typing.Union[BezierSimplex, ControlPoints, ControlPointsData]
    ] = None,
    fix: typing.Optional[typing.Iterable[Index]] = None,
    batch_size: typing.Optional[int] = None,
    max_epochs: typing.Optional[int] = None,
    accelerator: typing.Union[str, pl.accelerators.Accelerator] = "auto",
    strategy: typing.Union[str, pl.strategies.Strategy] = "auto",
    devices: typing.Union[typing.List[int], str, int] = "auto",
    num_nodes: int = 1,
    precision: typing.Union[str, int] = "32-true",
    enable_progress_bar: bool = False,
) -> BezierSimplex:
    r"""Fits a Bezier simplex.

    Parameters
    ----------
    params
        The data.
    values
        The label data.
    degree
        The degree of the Bezier simplex.
    init
        The initial values of a bezier simplex or control points.
    fix
        The indices of control points to exclude from training.
    batch_size
        The size of minibatch.
    max_epochs
        The number of epochs to stop training.
    accelerator
        The type of accelerators to use.
    strategy
        Distributed computing strategy.
    devices
        The number of accelerator devices to use.
    num_nodes
        The number of compute nodes to use.
    precision
        The precision of floating point numbers.
    enable_progress_bar
        Whether to enable progress bar.

    Returns
    -------
    A trained Bezier simplex.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf

    Prepare training data

    >>> ts = torch.tensor(  # parameters on a simplex
    ...     [
    ...         [3/3, 0/3, 0/3],
    ...         [2/3, 1/3, 0/3],
    ...         [2/3, 0/3, 1/3],
    ...         [1/3, 2/3, 0/3],
    ...         [1/3, 1/3, 1/3],
    ...         [1/3, 0/3, 2/3],
    ...         [0/3, 3/3, 0/3],
    ...         [0/3, 2/3, 1/3],
    ...         [0/3, 1/3, 2/3],
    ...         [0/3, 0/3, 3/3],
    ...     ]
    ... )
    >>> xs = 1 - ts * ts  # values corresponding to the parameters

    Train a model

    >>> bs = torch_bsf.fit(params=ts, values=xs, degree=3)

    Predict by the trained model

    >>> t = [[0.2, 0.3, 0.5]]
    >>> x = bs(t)
    >>> print(f"{t} -> {x}")
    [[0.2, 0.3, 0.5]] -> tensor([[0.9600, 0.9100, 0.7500]], grad_fn=<AddBackward0>)
    """
    data = TensorDataset(params, values)
    dl = DataLoader(data, batch_size=batch_size or len(data))

    if degree is None and init is None:
        raise ValueError("Either degree or init must be specified")
    if degree is not None and init is not None:
        raise ValueError("Either degree or init must be specified, not both")

    if isinstance(init, BezierSimplex):
        bs = init
    elif init is not None:
        bs = BezierSimplex(init)
    else:
        bs = randn(
            n_params=int(params.shape[1]),
            n_values=int(values.shape[1]),
            degree=typing.cast(int, degree),
        )

    fix = fix or []
    validate_simplex_indices(fix, bs.n_params, bs.degree)

    for index in fix:
        bs.control_points[index].requires_grad = False

    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        precision=precision,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="train_mse")],
        enable_progress_bar=enable_progress_bar,
    )
    trainer.fit(bs, dl)
    return bs
