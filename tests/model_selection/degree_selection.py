import subprocess
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from torch_bsf.model_selection.degree_selection import select_degree

_REPO_ROOT = Path(__file__).parent.parent.parent
_PARAMS_CSV = _REPO_ROOT / "params.csv"
_VALUES_CSV = _REPO_ROOT / "values.csv"
_CLI_TIMEOUT = 300  # seconds


def _make_simplex_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20, *, seed: int = 0):
    """Return (params, values) tensors on the standard simplex."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    raw = torch.rand(n_samples, n_params, generator=generator)
    params = raw / raw.sum(dim=1, keepdim=True)
    values = (1.0 - params ** 2)[:, :n_values]
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

    _PATCH = "torch_bsf.bezier_simplex._KFoldTrainer"

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


def _run_cli(*args, cwd=None):
    """Run `python -m torch_bsf.model_selection.degree_selection` with the given arguments."""
    workdir = cwd or _REPO_ROOT
    return subprocess.run(
        [sys.executable, "-m", "torch_bsf.model_selection.degree_selection"] + list(args),
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=_CLI_TIMEOUT,
    )


class TestDegreeSelectionCLI:
    def test_cli_basic_run(self, tmp_path):
        """CLI should complete successfully with minimal required arguments."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=2",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert any(line.startswith("Best degree:") for line in result.stdout.splitlines())

    def test_cli_output_is_valid_degree(self, tmp_path):
        """CLI stdout should contain 'Best degree: N' where N is in [min_degree, max_degree]."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=3",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        # "Best degree: N" may appear among other Lightning stdout lines
        matching = [line for line in result.stdout.splitlines() if line.startswith("Best degree:")]
        assert matching, f"'Best degree:' not found in stdout:\n{result.stdout}"
        degree = int(matching[0].split(":")[1].strip())
        assert 1 <= degree <= 3

    def test_cli_missing_params_fails(self):
        """CLI should exit non-zero when --params is missing."""
        result = _run_cli(f"--values={_VALUES_CSV}")
        assert result.returncode != 0

    def test_cli_missing_values_fails(self):
        """CLI should exit non-zero when --values is missing."""
        result = _run_cli(f"--params={_PARAMS_CSV}")
        assert result.returncode != 0

    def test_cli_inverted_degree_range_fails(self):
        """CLI should exit non-zero when --min_degree > --max_degree."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=5",
            "--max_degree=1",
        )
        assert result.returncode != 0
        assert "min_degree" in result.stderr.lower() or "max_degree" in result.stderr.lower()

    def test_cli_devices_auto(self, tmp_path):
        """CLI should accept --devices auto (string) without error."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--devices=auto",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_devices_integer(self, tmp_path):
        """CLI should accept --devices 1 (integer) without error."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--devices=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_loglevel_info_shows_progress(self, tmp_path):
        """With --loglevel INFO, degree progress messages should appear in stderr."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=INFO",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "Checking degree" in result.stderr

    def test_cli_loglevel_warning_suppresses_progress(self, tmp_path):
        """With --loglevel WARNING, degree progress messages should not appear."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "Checking degree" not in result.stderr

    def test_cli_help(self):
        """CLI should print help and exit 0."""
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--params" in result.stdout
        assert "--loglevel" in result.stdout
        assert "--devices" in result.stdout
        assert "--patience" in result.stdout


class TestSelectDegreePatience:
    """Tests for the patience-based early-stopping logic in select_degree."""

    _PATCH = "torch_bsf.bezier_simplex._KFoldTrainer"

    def _select_with_mse_sequence(self, mse_sequence, patience=1, min_degree=1):
        """Run select_degree with a predetermined sequence of per-degree MSE values."""
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        mse_iter = iter(mse_sequence)

        def make_trainer(**kwargs):
            mock = MagicMock()
            mse = next(mse_iter)
            mock.cross_validate.return_value = [[{"test_mse": mse}]]
            return mock

        max_degree = min_degree + len(mse_sequence) - 1
        with patch(self._PATCH, side_effect=make_trainer):
            return select_degree(
                params,
                values,
                min_degree=min_degree,
                max_degree=max_degree,
                num_folds=2,
                patience=patience,
            )

    def test_patience_1_stops_after_first_increase(self):
        """With patience=1, stops as soon as MSE increases."""
        # MSE decreases at degree 2, increases at degree 3 → should return 2
        best = self._select_with_mse_sequence([0.5, 0.3, 0.4], patience=1)
        assert best == 2

    def test_patience_2_tolerates_one_increase(self):
        """With patience=2, one transient increase is tolerated."""
        # MSE: 0.5, 0.3, 0.35 (up), 0.2 (new best) → should return 4
        best = self._select_with_mse_sequence([0.5, 0.3, 0.35, 0.2], patience=2, min_degree=1)
        assert best == 4

    def test_patience_2_stops_after_two_increases(self):
        """With patience=2, two consecutive increases trigger early stop."""
        # MSE: 0.5, 0.3, 0.35 (up 1), 0.4 (up 2) → stops, best is 2
        best = self._select_with_mse_sequence([0.5, 0.3, 0.35, 0.4], patience=2, min_degree=1)
        assert best == 2

    def test_patience_high_traverses_all_degrees(self):
        """Large patience causes all degrees to be checked."""
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        call_count = 0

        def make_trainer(**kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.cross_validate.return_value = [[{"test_mse": 1.0 + call_count}]]
            return mock

        with patch(self._PATCH, side_effect=make_trainer):
            select_degree(
                params, values,
                min_degree=1, max_degree=4,
                num_folds=2,
                patience=10,
            )
        # All 4 degrees (1..4) should have been evaluated
        assert call_count == 4

    def test_patience_zero_raises(self):
        """patience=0 is invalid and should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="patience"):
            select_degree(params, values, patience=0)

    def test_patience_negative_raises(self):
        """Negative patience is invalid and should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="patience"):
            select_degree(params, values, patience=-1)


class TestSelectDegreeEdgeCases:
    """Tests for edge-case validation in select_degree."""

    def test_empty_dataset_raises(self):
        """select_degree with zero samples should raise ValueError."""
        params = torch.empty(0, 3)
        values = torch.empty(0, 2)
        with pytest.raises(ValueError, match="non-empty"):
            select_degree(params, values, min_degree=1, max_degree=2, num_folds=2)

    def test_invalid_batch_size_raises(self):
        """Passing a non-positive batch_size should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="batch_size"):
            select_degree(params, values, min_degree=1, max_degree=1, num_folds=2, batch_size=-1)


class TestDegreeSelectionCliMain:
    """Tests for the _cli_main() entry point of degree_selection."""

    def test_cli_main_basic(self, tmp_path, monkeypatch):
        """_cli_main() should run successfully with minimal arguments."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--min_degree=1",
                "--max_degree=2",
                "--num_folds=2",
                "--max_epochs=1",
                "--loglevel=WARNING",
            ],
        )
        _cli_main()

    def test_cli_main_with_batch_size(self, tmp_path, monkeypatch):
        """_cli_main() should accept --batch_size."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--min_degree=1",
                "--max_degree=1",
                "--num_folds=2",
                "--max_epochs=1",
                "--batch_size=10",
                "--loglevel=WARNING",
            ],
        )
        _cli_main()


