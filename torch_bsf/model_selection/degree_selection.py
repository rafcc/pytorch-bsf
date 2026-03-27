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
    **trainer_kwargs
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
    trainer_kwargs
        Additional arguments forwarded to :class:`~pl_crossvalidate.KFoldTrainer`.
        By default, ``limit_val_batches=0.0`` and ``num_sanity_val_steps=0`` are
        set so that no validation loop is run inside each fold — the
        cross-validation estimate is taken from ``test_mse`` (the held-out test
        subset produced by KFoldTrainer after each fold's ``fit`` call).
        If you need a validation loop within each fold (e.g., for
        ``EarlyStopping``), pass ``limit_val_batches=1.0`` and supply
        ``val_dataloaders`` via a custom
        :class:`~pl_crossvalidate.KFoldDataModule`.

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
    batch_size = trainer_kwargs.pop("batch_size", len(dataset))
    train_dl = DataLoader(dataset, batch_size=batch_size)

    # Disable validation monitoring during training; the CV estimate comes from
    # test_step (the per-fold held-out subset produced by KFoldTrainer).
    # Callers can override these defaults via trainer_kwargs.
    kfold_kwargs: dict = {"limit_val_batches": 0.0, "num_sanity_val_steps": 0}
    kfold_kwargs.update(trainer_kwargs)

    best_degree = min_degree
    best_mse = float('inf')

    for d in range(min_degree, max_degree + 1):
        _logger.info("Checking degree %d...", d)

        model = randn(params.shape[1], values.shape[1], d)

        trainer = KFoldTrainer(num_folds=num_folds, **kfold_kwargs)

        # KFoldTrainer splits train_dl into per-fold train/test subsets;
        # test results (test_mse) give the unbiased cross-validation estimate.
        stats = trainer.cross_validate(model, train_dataloader=train_dl)

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
