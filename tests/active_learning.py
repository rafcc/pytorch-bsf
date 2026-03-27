import pytest
import torch

import torch_bsf.bezier_simplex as tbbs
from torch_bsf.active_learning import suggest_next_points


def _make_models(n_params: int, n_values: int, degree: int, k: int = 2):
    """Return a list of k random BezierSimplex models."""
    return [tbbs.randn(n_params=n_params, n_values=n_values, degree=degree) for _ in range(k)]


class TestSuggestNextPointsQBC:
    def test_output_shape(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=3, n_candidates=50)
        assert result.shape == (3, 3)

    def test_output_is_on_simplex(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=5, n_candidates=100)
        # All components non-negative
        assert (result >= 0).all()
        # Rows sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(5), atol=1e-5)

    def test_single_suggestion(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=1, n_candidates=50)
        assert result.shape == (1, 3)

    def test_single_model(self):
        # With one model, QBC variance is zero everywhere; should still return a valid point
        model = tbbs.randn(n_params=3, n_values=2, degree=2)
        result = suggest_next_points([model], n_suggestions=1, n_candidates=50)
        assert result.shape == (1, 3)
        assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            suggest_next_points([])

    def test_mismatched_n_params_raises(self):
        m1 = tbbs.randn(n_params=3, n_values=2, degree=2)
        m2 = tbbs.randn(n_params=2, n_values=2, degree=2)
        with pytest.raises(ValueError, match="n_params"):
            suggest_next_points([m1, m2])

    def test_unknown_method_raises(self):
        models = _make_models(3, 2, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            suggest_next_points(models, method="unknown")


class TestSuggestNextPointsDensity:
    def test_output_shape(self):
        models = _make_models(3, 2, 2)
        existing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = suggest_next_points(
            models, n_suggestions=2, n_candidates=50, method="density", params=existing
        )
        assert result.shape == (2, 3)

    def test_output_is_on_simplex(self):
        models = _make_models(3, 2, 2)
        existing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = suggest_next_points(
            models, n_suggestions=3, n_candidates=100, method="density", params=existing
        )
        assert (result >= 0).all()
        assert torch.allclose(result.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_density_requires_params(self):
        models = _make_models(3, 2, 2)
        with pytest.raises(ValueError, match="params"):
            suggest_next_points(models, method="density", params=None)
