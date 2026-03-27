import numpy as np
import pytest

pytest.importorskip("sklearn")

from torch_bsf.sklearn import BezierSimplexRegressor


def _make_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20):
    """Return (X, y) on the standard simplex with output dimension n_values."""
    rng = np.random.default_rng(42)
    raw = rng.random((n_samples, n_params))
    X = raw / raw.sum(axis=1, keepdims=True)
    y = (1.0 - X ** 2)[:, :n_values]
    return X.astype(np.float32), y.astype(np.float32)


class TestBezierSimplexRegressor:
    def test_fit_predict(self):
        X, y = _make_data()
        reg = BezierSimplexRegressor(
            degree=2, max_epochs=5, accelerator="cpu", devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_score_returns_float(self):
        X, y = _make_data()
        reg = BezierSimplexRegressor(
            degree=2, max_epochs=5, accelerator="cpu", devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y)
        s = reg.score(X, y)
        assert isinstance(s, float)

    def test_predict_requires_fit(self):
        from sklearn.exceptions import NotFittedError

        reg = BezierSimplexRegressor(degree=2)
        X, _ = _make_data()
        with pytest.raises(NotFittedError):
            reg.predict(X)

    def test_get_set_params(self):
        reg = BezierSimplexRegressor(degree=3, max_epochs=50)
        params = reg.get_params()
        assert params["degree"] == 3
        assert params["max_epochs"] == 50
        reg.set_params(degree=2)
        assert reg.degree == 2

    def test_sklearn_interface_compliance(self):
        """Verify that BezierSimplexRegressor satisfies the core sklearn estimator interface."""
        from sklearn.base import clone

        reg = BezierSimplexRegressor(degree=2, max_epochs=5)
        # clone() relies on get_params() / __init__ signature parity
        cloned = clone(reg)
        assert cloned.degree == reg.degree
        assert cloned.max_epochs == reg.max_epochs
        # Cloned estimator should not share state
        assert cloned is not reg
