import torch

from torch_bsf.model_selection.degree_selection import select_degree


def _make_simplex_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20, *, seed: int = 0):
    """Return (params, values) tensors on the standard simplex."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    raw = torch.rand(n_samples, n_params, generator=generator)
    params = raw / raw.sum(dim=1, keepdim=True)
    values = 1.0 - params ** 2
    return params, values


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
