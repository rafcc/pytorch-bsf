"""Basic benchmarks for torch_bsf.

Run with::

    pytest tests/benchmark.py --benchmark-only

These benchmarks are intentionally skipped in the normal ``pytest`` run so
they do not slow down the unit-test suite.  They are useful for detecting
performance regressions in the forward pass and in :func:`torch_bsf.fit`.
"""

import pytest
import torch

pytest.importorskip("pytest_benchmark")

import torch_bsf
import torch_bsf.bezier_simplex as tbbs
from torch_bsf.sampling import simplex_grid


# ---------------------------------------------------------------------------
# Forward-pass benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="forward")
@pytest.mark.parametrize(
    "n_params, n_values, degree, batch_size",
    [
        (3, 3, 3, 100),
        (3, 3, 5, 100),
        (4, 4, 3, 100),
        (3, 3, 3, 1000),
    ],
)
def test_benchmark_forward(n_params, n_values, degree, batch_size, benchmark):
    """Benchmark the BezierSimplex forward pass."""
    bs = tbbs.zeros(n_params=n_params, n_values=n_values, degree=degree)
    pts = simplex_grid(n_params=n_params, degree=degree)[:batch_size]
    if len(pts) < batch_size:
        # Repeat to reach batch_size
        repeats = (batch_size + len(pts) - 1) // len(pts)
        pts = pts.repeat(repeats, 1)[:batch_size]

    def _forward():
        with torch.no_grad():
            return bs(pts)

    benchmark(_forward)


# ---------------------------------------------------------------------------
# fit() benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="fit")
def test_benchmark_fit(benchmark):
    """Benchmark a short training run of torch_bsf.fit()."""
    params = simplex_grid(n_params=3, degree=3)
    values = params.pow(2).sum(dim=1, keepdim=True)

    fast = dict(
        max_epochs=5,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )

    def _fit():
        return torch_bsf.fit(params=params, values=values, degree=2, **fast)

    benchmark(_fit)
