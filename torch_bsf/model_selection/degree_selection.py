import torch
from pl_crossvalidate import KFoldTrainer

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
    from torch.utils.data import TensorDataset

    best_degree = min_degree
    best_mse = float('inf')

    for d in range(min_degree, max_degree + 1):
        print(f"Checking degree {d}...")
        
        # We need a model and data for KFoldTrainer
        # KFoldTrainer works on a model and a datamodule
        # But we can also pass a dataset
        dataset = TensorDataset(params, values)
        
        # Setup dummy model to get dimensions
        model = randn(params.shape[1], values.shape[1], d)
        
        trainer = KFoldTrainer(
            num_folds=num_folds,
            **trainer_kwargs
        )
        
        # We want to measure val_mse
        stats = trainer.cross_validate(model, dataset=dataset)
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
        print(f"Degree {d}: Mean MSE = {mean_mse:.6f}")
        
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = d
        else:
            # Simple heuristic: if MSE increases, stop
            if d > min_degree + 1:
                print(f"MSE increased at degree {d}, stopping.")
                break
                
    return best_degree
