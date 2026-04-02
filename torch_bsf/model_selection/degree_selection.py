import logging
import torch
from pl_crossvalidate import KFoldTrainer

_logger = logging.getLogger(__name__)

def select_degree(
    params: torch.Tensor,
    values: torch.Tensor,
    min_degree: int = 1,
    max_degree: int = 5,
    num_folds: int = 5,
    val_dataloaders=None,
    datamodule=None,
    **trainer_kwargs
) -> int:
    """Select the best degree for the Bézier simplex using cross-validation.

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
        The best degree found.
    """
    from torch_bsf.bezier_simplex import randn
    from torch.utils.data import DataLoader, TensorDataset

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
        train_dl = DataLoader(dataset, batch_size=batch_size)
    else:
        train_dl = None  # datamodule provides its own dataloaders

    # Disable validation monitoring during training by default; the CV estimate
    # comes from test_step (the per-fold held-out subset produced by KFoldTrainer).
    # When the caller supplies val_dataloaders or a datamodule, respect those by
    # not forcing limit_val_batches=0.0.
    kfold_kwargs: dict = {"num_sanity_val_steps": 0}
    if val_dataloaders is None and datamodule is None:
        kfold_kwargs["limit_val_batches"] = 0.0
    kfold_kwargs.update(trainer_kwargs)

    best_degree = min_degree
    best_mse = float('inf')

    for d in range(min_degree, max_degree + 1):
        _logger.info("Checking degree %d...", d)

        model = randn(params.shape[1], values.shape[1], d)

        trainer = KFoldTrainer(num_folds=num_folds, **kfold_kwargs)

        # KFoldTrainer splits train_dl into per-fold train/test subsets;
        # test results (test_mse) give the unbiased cross-validation estimate.
        cross_validate_kwargs: dict = {}
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

        mean_mse = sum(test_mses) / len(test_mses) if test_mses else float('inf')
        _logger.info("Degree %d: Mean MSE = %.6f", d, mean_mse)
        
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = d
        else:
            # Simple heuristic: if MSE increases, stop
            if d > min_degree + 1:
                _logger.info("MSE increased at degree %d, stopping.", d)
                break
                
    return best_degree
