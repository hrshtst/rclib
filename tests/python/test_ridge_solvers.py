"""Tests for different Ridge Regression solvers."""

from __future__ import annotations

import numpy as np
import pytest
from rclib import ESN, readouts, reservoirs


@pytest.mark.parametrize("solver", ["cholesky", "conjugate_gradient", "conjugate_gradient_implicit"])
def test_ridge_solvers(solver: str) -> None:
    """Test that all three Ridge solvers produce accurate results on a simple task."""
    # 1. Prepare data
    # Simple linear problem: y = 2x + 1
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * x + 1

    # 2. Configure Reservoir (Small to make tests fast)
    res = reservoirs.RandomSparse(n_neurons=50, spectral_radius=0.9, sparsity=0.1, seed=42)

    # 3. Configure Readout with specific solver
    readout = readouts.Ridge(alpha=1e-6, include_bias=True, solver=solver)

    # 4. Build Model
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # 5. Train
    model.fit(x, y)

    # 6. Predict
    y_pred = model.predict(x)

    # 7. Check Error
    mse = float(np.mean((y_pred - y) ** 2))

    # All solvers should be able to solve this simple problem very accurately
    mse_threshold = 1e-2
    assert mse < mse_threshold, f"Solver {solver} failed with MSE: {mse}"


def test_invalid_solver() -> None:
    """Test that an unsupported solver name raises a ValueError."""
    readout = readouts.Ridge(alpha=1e-6, include_bias=True, solver="invalid_solver_name")
    model = ESN()
    with pytest.raises(ValueError, match="Unsupported solver"):
        model.set_readout(readout)
