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
    # Manual setup since fit() or DataModule usually expects files
    # We can create a SimpleDataModule or use TensorDataset
    from torch.utils.data import DataLoader, TensorDataset

    best_degree = min_degree
    best_mse = float('inf')

    for d in range(min_degree, max_degree + 1):
        _logger.info("Checking degree %d...", d)
        
        # We need a model and data for KFoldTrainer
        # KFoldTrainer works on a model and a dataloader
        dataset = TensorDataset(params, values)
        train_dl = DataLoader(dataset, batch_size=len(params))
        
        # Setup dummy model to get dimensions
        model = randn(params.shape[1], values.shape[1], d)
        
        trainer = KFoldTrainer(
            num_folds=num_folds,
            **trainer_kwargs
        )
        
        # We want to measure val_mse
        # Use same data for validation (matches split_ratio=1.0 in BezierSimplexDataModule)
        stats = trainer.cross_validate(model, train_dataloader=train_dl, val_dataloaders=train_dl)
        # stats is a list of results for each fold
        # Find the mean validation MSE
        val_mses = []
        for fold_results in stats:
            for res in fold_results:
                if 'val_mse' in res:
                    val_mses.append(res['val_mse'])
        
        if not val_mses:
            # Fallback to training MSE if validation MSE is not logged
            for fold_results in stats:
                for res in fold_results:
                    if 'train_mse' in res:
                        val_mses.append(res['train_mse'])

        mean_mse = sum(val_mses) / len(val_mses) if val_mses else float('inf')
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
