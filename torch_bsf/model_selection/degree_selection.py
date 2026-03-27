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
        Additional arguments for the KFoldTrainer and Trainer.

    Returns
    -------
        The best degree found.
    """
    from torch_bsf.bezier_simplex import randn
    from torch.utils.data import DataLoader, TensorDataset

    # Build the dataset once – it doesn't change across degree iterations
    dataset = TensorDataset(params, values)
    train_dl = DataLoader(dataset)

    # Disable validation monitoring during training; the CV estimate comes from
    # test_step (the per-fold held-out subset produced by KFoldTrainer).
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
