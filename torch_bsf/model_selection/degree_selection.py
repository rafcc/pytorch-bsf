import logging
from typing import Any

import lightning.pytorch as L
import torch
from pl_crossvalidate import KFoldTrainer
from torch.utils.data import DataLoader, TensorDataset

_logger = logging.getLogger(__name__)


def select_degree(
    params: torch.Tensor,
    values: torch.Tensor,
    min_degree: int = 1,
    max_degree: int = 5,
    num_folds: int = 5,
    patience: int = 1,
    val_dataloaders: Any | None = None,
    datamodule: L.LightningDataModule | None = None,
    **trainer_kwargs: Any,
) -> int:
    """Select the best degree for the Bezier simplex using cross-validation.

    Parameters
    ----------
    params
        The input parameters.
    values
        The output values.
    min_degree
        Starting degree to check.
    max_degree
        Ending degree to check.
    num_folds
        Number of folds for cross-validation.
    patience
        Number of consecutive degrees without improvement before early
        stopping.  Defaults to ``1``, which stops as soon as the mean CV
        MSE increases for the first time (after ``min_degree + 1``).
        Increase this value to tolerate transient upswings caused by
        numerical noise.
    val_dataloaders
        Optional validation dataloader(s) forwarded to
        :meth:`~pl_crossvalidate.KFoldTrainer.cross_validate` for fold-internal
        validation (e.g., to use ``EarlyStopping(monitor="val_mse")``).  When
        provided, ``limit_val_batches`` is **not** forced to ``0.0``.
    datamodule
        Optional Lightning DataModule forwarded to
        :meth:`~pl_crossvalidate.KFoldTrainer.cross_validate`.  When a
        ``datamodule`` is provided the ``train_dataloader`` built internally is
        **not** used.
    trainer_kwargs
        Additional arguments forwarded to :class:`~pl_crossvalidate.KFoldTrainer`,
        except for the special-case ``batch_size`` key.  If supplied,
        ``batch_size`` is consumed here to set the batch size of the
        ``DataLoader`` used for cross-validation and is **not** forwarded to
        :class:`~pl_crossvalidate.KFoldTrainer`.  By default, full-batch loading
        (``batch_size=len(dataset)``) is used for consistency with
        :func:`torch_bsf.bezier_simplex.fit`.

        By default, ``limit_val_batches=0.0`` and ``num_sanity_val_steps=0`` are
        set so that no validation loop is run inside each fold — the
        cross-validation estimate is taken from ``test_mse`` (the held-out test
        subset produced by KFoldTrainer after each fold's ``fit`` call).
        If you need a validation loop within each fold (e.g., for
        ``EarlyStopping``), pass ``val_dataloaders`` or ``datamodule``;
        ``limit_val_batches`` will then be left at its default value of ``1.0``.

    Returns
    -------
    int
        The best degree found.
    """
    from torch_bsf.bezier_simplex import randn

    # Build the dataset once – it doesn't change across degree iterations.
    # Use full-batch loading (consistent with bezier_simplex.fit()) unless the
    # caller explicitly provides a different batch_size via trainer_kwargs.
    dataset = TensorDataset(params, values)
    if len(dataset) == 0:
        raise ValueError(
            "select_degree requires a non-empty dataset; got 0 samples. "
            "Ensure that 'params' and 'values' contain at least one example."
        )
    # batch_size is always popped from trainer_kwargs to prevent it from being
    # forwarded to KFoldTrainer (which does not accept it).  When a datamodule is
    # supplied the caller's own data pipeline controls batching, so we only
    # validate and use batch_size when we need to build train_dl ourselves.
    batch_size = trainer_kwargs.pop("batch_size", None)
    if datamodule is None:
        if batch_size is None:
            batch_size = len(dataset)
        elif not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer or None, got {batch_size!r}. "
                "Either provide a positive 'batch_size' in trainer_kwargs or omit it "
                "to use the default (full-batch)."
            )
        train_dl: DataLoader | None = DataLoader(dataset, batch_size=batch_size)
    else:
        train_dl = None  # datamodule provides its own dataloaders

    # Disable validation monitoring during training by default; the CV estimate
    # comes from test_step (the per-fold held-out subset produced by KFoldTrainer).
    # When the caller supplies val_dataloaders or a datamodule, respect those by
    # not forcing limit_val_batches=0.0.
    kfold_kwargs: dict[str, Any] = {"num_sanity_val_steps": 0}
    if val_dataloaders is None and datamodule is None:
        kfold_kwargs["limit_val_batches"] = 0.0
    kfold_kwargs.update(trainer_kwargs)

    best_degree = min_degree
    best_mse = float("inf")
    no_improve_count = 0

    for d in range(min_degree, max_degree + 1):
        _logger.info("Checking degree %d...", d)

        model = randn(params.shape[1], values.shape[1], d)

        trainer = KFoldTrainer(num_folds=num_folds, **kfold_kwargs)

        # KFoldTrainer splits train_dl into per-fold train/test subsets;
        # test results (test_mse) give the unbiased cross-validation estimate.
        cross_validate_kwargs: dict[str, Any] = {}
        if datamodule is not None:
            cross_validate_kwargs["datamodule"] = datamodule
        else:
            cross_validate_kwargs["train_dataloader"] = train_dl
        if val_dataloaders is not None:
            cross_validate_kwargs["val_dataloaders"] = val_dataloaders
        stats = trainer.cross_validate(model, **cross_validate_kwargs)

        test_mses = [
            res["test_mse"]
            for fold_results in stats
            for res in fold_results
            if "test_mse" in res
        ]

        mean_mse = sum(test_mses) / len(test_mses) if test_mses else float("inf")
        _logger.info("Degree %d: Mean MSE = %.6f", d, mean_mse)

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = d
            no_improve_count = 0
        else:
            no_improve_count += 1
            if d > min_degree and no_improve_count >= patience:
                _logger.info(
                    "No improvement for %d consecutive degree(s) at degree %d, stopping.",
                    no_improve_count,
                    d,
                )
                break

    return best_degree


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np

    from torch_bsf.validator import int_or_str

    parser = ArgumentParser(
        prog="python -m torch_bsf.model_selection.degree_selection",
        description="Automatic degree selection for Bezier simplex via k-fold cross-validation",
    )
    parser.add_argument("--params", type=Path, required=True, help="Path to the input parameters CSV file")
    parser.add_argument("--values", type=Path, required=True, help="Path to the output values CSV file")
    parser.add_argument("--header", type=int, default=0, help="Number of header rows to skip (default: 0)")
    parser.add_argument("--min_degree", type=int, default=1, help="Minimum degree to search (default: 1)")
    parser.add_argument("--max_degree", type=int, default=5, help="Maximum degree to search (default: 5)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument(
        "--patience",
        type=int,
        default=1,
        help="Consecutive non-improving degrees before stopping (default: 1)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (default: full-batch)")
    parser.add_argument("--max_epochs", type=int, default=2, help="Training epochs per fold (default: 2)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (default: auto)")
    parser.add_argument("--devices", type=int_or_str, default="auto", help="Devices to use, integer or 'auto' (default: auto)")
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level for degree selection progress (default: INFO)",
    )
    args = parser.parse_args()

    if args.min_degree > args.max_degree:
        parser.error(f"--min_degree ({args.min_degree}) must be <= --max_degree ({args.max_degree})")

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )

    def _load_csv(path: Path, header: int) -> torch.Tensor:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        return torch.from_numpy(
            np.loadtxt(path, delimiter=delimiter, skiprows=header, ndmin=2)
        ).to(torch.get_default_dtype())

    params_tensor = _load_csv(args.params, args.header)
    values_tensor = _load_csv(args.values, args.header)

    trainer_kwargs_cli: dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    if args.batch_size is not None:
        trainer_kwargs_cli["batch_size"] = args.batch_size

    best_degree = select_degree(
        params=params_tensor,
        values=values_tensor,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
        num_folds=args.num_folds,
        patience=args.patience,
        **trainer_kwargs_cli,
    )

    print(f"Best degree: {best_degree}")
