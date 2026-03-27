import torch
from unittest.mock import MagicMock, patch

from torch_bsf.model_selection.degree_selection import select_degree


def _make_simplex_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20, *, seed: int = 0):
    """Return (params, values) tensors on the standard simplex."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    raw = torch.rand(n_samples, n_params, generator=generator)
    params = raw / raw.sum(dim=1, keepdim=True)
    values = 1.0 - params ** 2
    return params, values


def _make_mock_trainer(test_mse: float = 0.1):
    """Return a MagicMock KFoldTrainer that yields a single fold result."""
    mock = MagicMock()
    mock.cross_validate.return_value = [[{"test_mse": test_mse}]]
    return mock


class TestSelectDegree:
    def test_returns_int_in_range(self):
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        best = select_degree(
            params, values,
            min_degree=1, max_degree=3,
            num_folds=2,
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        assert isinstance(best, int)
        assert 1 <= best <= 3

    def test_min_equals_max(self):
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        best = select_degree(
            params, values,
            min_degree=2, max_degree=2,
            num_folds=2,
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        assert best == 2


class TestSelectDegreeKwargForwarding:
    """Verify correct kwargs are forwarded to KFoldTrainer.cross_validate."""

    _PATCH = "torch_bsf.model_selection.degree_selection.KFoldTrainer"

    def test_default_forces_limit_val_batches(self):
        """Without val_dataloaders/datamodule, limit_val_batches=0.0 is set."""
        params, values = _make_simplex_data()
        mock_trainer = _make_mock_trainer()

        with patch(self._PATCH, return_value=mock_trainer) as mock_cls:
            select_degree(params, values, min_degree=1, max_degree=1, num_folds=2)

        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("limit_val_batches") == 0.0

        cv_kwargs = mock_trainer.cross_validate.call_args.kwargs
        assert "train_dataloader" in cv_kwargs
        assert "datamodule" not in cv_kwargs
        assert "val_dataloaders" not in cv_kwargs

    def test_val_dataloaders_suppresses_limit_val_batches(self):
        """Passing val_dataloaders does not force limit_val_batches=0.0."""
        params, values = _make_simplex_data()
        mock_trainer = _make_mock_trainer()
        val_dl = MagicMock()

        with patch(self._PATCH, return_value=mock_trainer) as mock_cls:
            select_degree(
                params, values, min_degree=1, max_degree=1, num_folds=2,
                val_dataloaders=val_dl,
            )

        init_kwargs = mock_cls.call_args.kwargs
        assert "limit_val_batches" not in init_kwargs

        cv_kwargs = mock_trainer.cross_validate.call_args.kwargs
        assert cv_kwargs.get("val_dataloaders") is val_dl
        assert "train_dataloader" in cv_kwargs
        assert "datamodule" not in cv_kwargs

    def test_datamodule_suppresses_train_dataloader(self):
        """Passing datamodule disables internal train_dataloader and limit_val_batches=0.0."""
        params, values = _make_simplex_data()
        mock_trainer = _make_mock_trainer()
        dm = MagicMock()

        with patch(self._PATCH, return_value=mock_trainer) as mock_cls:
            select_degree(
                params, values, min_degree=1, max_degree=1, num_folds=2,
                datamodule=dm,
            )

        init_kwargs = mock_cls.call_args.kwargs
        assert "limit_val_batches" not in init_kwargs

        cv_kwargs = mock_trainer.cross_validate.call_args.kwargs
        assert cv_kwargs.get("datamodule") is dm
        assert "train_dataloader" not in cv_kwargs

    def test_batch_size_ignored_with_datamodule(self):
        """batch_size in trainer_kwargs is silently consumed when datamodule is provided."""
        params, values = _make_simplex_data()
        mock_trainer = _make_mock_trainer()
        dm = MagicMock()

        with patch(self._PATCH, return_value=mock_trainer):
            # Should not raise even though batch_size has no effect here.
            select_degree(
                params, values, min_degree=1, max_degree=1, num_folds=2,
                datamodule=dm, batch_size=16,
            )
