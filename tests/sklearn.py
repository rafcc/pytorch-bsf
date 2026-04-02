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


class TestSetParams:
    def test_set_params_returns_self(self):
        """set_params() must return self to support method chaining."""
        reg = BezierSimplexRegressor(degree=3)
        result = reg.set_params(degree=2)
        assert result is reg

    def test_set_params_updates_multiple_params(self):
        """set_params() updates several attributes in one call."""
        reg = BezierSimplexRegressor(degree=3, max_epochs=100, smoothness_weight=0.1)
        reg.set_params(degree=1, max_epochs=10, smoothness_weight=0.5)
        assert reg.degree == 1
        assert reg.max_epochs == 10
        assert reg.smoothness_weight == 0.5

    def test_set_params_reflected_in_get_params(self):
        """get_params() reflects values changed by set_params()."""
        reg = BezierSimplexRegressor(degree=3, max_epochs=50)
        reg.set_params(degree=1)
        params = reg.get_params()
        assert params["degree"] == 1
        assert params["max_epochs"] == 50

    def test_set_params_reflected_in_fit(self):
        """After set_params(), fit trains with the updated degree."""
        X, y = _make_data()
        reg = BezierSimplexRegressor(
            degree=3, max_epochs=3, accelerator="cpu", devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.set_params(degree=1)
        reg.fit(X, y)
        assert reg.model_.degree == 1

    def test_set_params_invalid_key_raises(self):
        """set_params() with an unknown parameter name raises ValueError."""
        reg = BezierSimplexRegressor(degree=3)
        with pytest.raises(ValueError):
            reg.set_params(nonexistent_param=42)


class TestPipelineIntegration:
    _kwargs = dict(
        degree=2, max_epochs=3, accelerator="cpu", devices=1,
        trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
    )

    def test_pipeline_fit_predict(self):
        """BezierSimplexRegressor works as the final step of a Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        def to_simplex(X):
            return X / X.sum(axis=1, keepdims=True)

        rng = np.random.default_rng(0)
        X_raw = rng.random((20, 3)).astype(np.float32)
        _, y = _make_data()

        pipe = Pipeline([
            ("normalize", FunctionTransformer(to_simplex)),
            ("reg", BezierSimplexRegressor(**self._kwargs)),
        ])
        pipe.fit(X_raw, y)
        preds = pipe.predict(X_raw)
        assert preds.shape == y.shape

    def test_pipeline_score(self):
        """Pipeline.score() delegates to BezierSimplexRegressor.score()."""
        from sklearn.pipeline import Pipeline

        X, y = _make_data()
        pipe = Pipeline([("reg", BezierSimplexRegressor(**self._kwargs))])
        pipe.fit(X, y)
        s = pipe.score(X, y)
        assert isinstance(s, float)

    def test_pipeline_set_params(self):
        """Pipeline.set_params() propagates to BezierSimplexRegressor via step__param notation."""
        from sklearn.pipeline import Pipeline

        X, y = _make_data()
        pipe = Pipeline([("reg", BezierSimplexRegressor(**self._kwargs))])
        pipe.set_params(reg__degree=1)
        assert pipe.named_steps["reg"].degree == 1
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == y.shape
        assert pipe.named_steps["reg"].model_.degree == 1

    def test_pipeline_clone(self):
        """clone() on a Pipeline containing BezierSimplexRegressor is safe."""
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("reg", BezierSimplexRegressor(**self._kwargs))])
        cloned = clone(pipe)
        assert cloned.named_steps["reg"].degree == pipe.named_steps["reg"].degree
        assert cloned is not pipe


class TestGridSearchCVIntegration:
    _kwargs = dict(
        max_epochs=3, accelerator="cpu", devices=1,
        trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
    )

    def test_gridsearchcv_direct(self):
        """GridSearchCV finds a best degree when searching over BezierSimplexRegressor."""
        from sklearn.model_selection import GridSearchCV

        X, y = _make_data(n_samples=30)
        reg = BezierSimplexRegressor(**self._kwargs)
        gs = GridSearchCV(reg, {"degree": [1, 2]}, cv=2, scoring="r2", refit=True, n_jobs=1)
        gs.fit(X, y)
        assert gs.best_params_["degree"] in [1, 2]
        preds = gs.predict(X)
        assert preds.shape == y.shape

    def test_gridsearchcv_in_pipeline(self):
        """GridSearchCV searches over BezierSimplexRegressor parameters inside a Pipeline."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        X, y = _make_data(n_samples=30)
        pipe = Pipeline([("reg", BezierSimplexRegressor(**self._kwargs))])
        gs = GridSearchCV(
            pipe, {"reg__degree": [1, 2]}, cv=2, scoring="r2", refit=True, n_jobs=1,
        )
        gs.fit(X, y)
        assert gs.best_params_["reg__degree"] in [1, 2]
        preds = gs.predict(X)
        assert preds.shape == y.shape

    def test_gridsearchcv_cv_results(self):
        """GridSearchCV produces expected cv_results_ structure."""
        from sklearn.model_selection import GridSearchCV

        X, y = _make_data(n_samples=30)
        reg = BezierSimplexRegressor(**self._kwargs)
        gs = GridSearchCV(reg, {"degree": [1, 2]}, cv=2, scoring="r2", n_jobs=1)
        gs.fit(X, y)
        assert "mean_test_score" in gs.cv_results_
        assert len(gs.cv_results_["params"]) == 2
