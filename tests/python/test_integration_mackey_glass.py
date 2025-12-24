"""Integration and performance tests for rclib."""

from __future__ import annotations

import time

import numpy as np
import pytest
from rclib import ESN, readouts, reservoirs


def mackey_glass(n_samples: int = 1500, tau: int = 17, seed: int = 0) -> np.ndarray:
    """Deterministic Mackey-Glass for testing."""
    rng = np.random.default_rng(seed=seed)
    x = np.zeros(n_samples + tau)
    x[0:tau] = 0.5 + 0.5 * rng.random(tau)
    for t in range(tau, n_samples + tau - 1):
        x[t + 1] = x[t] + (0.2 * x[t - tau]) / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    return x[tau:]


def test_mackey_glass_integration_accuracy() -> None:
    """Verify that ESN achieves expected accuracy on Mackey-Glass task."""
    # 1. Setup deterministic data
    data = mackey_glass(n_samples=1500, seed=42)
    x_data = data[:-1].reshape(-1, 1)
    y_data = data[1:].reshape(-1, 1)

    train_len = 1000
    washout_len = 100
    x_train, y_train = x_data[:train_len], y_data[:train_len]
    x_test, y_test = x_data[train_len:], y_data[train_len:]

    # 2. Configure Model (Identical to example)
    res = reservoirs.RandomSparse(
        n_neurons=1000,
        spectral_radius=1.1,
        sparsity=0.05,
        leak_rate=0.1,
        include_bias=True,
        input_scaling=0.5,
    )
    readout = readouts.Ridge(alpha=1e-8, include_bias=True)

    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # 3. Fit and Predict
    model.fit(x_train, y_train, washout_len)
    y_pred = model.predict(x_test, reset_state_before_predict=False)

    # 4. Assertions
    mse = np.mean((y_pred - y_test) ** 2)

    # We expect MSE to be around 2.0e-4.
    # We set a slightly higher bound to allow for minor numerical variations across architectures.
    expected_mse_upper_bound = 2.5e-4

    assert mse < expected_mse_upper_bound, f"MSE {mse:.4e} exceeded threshold {expected_mse_upper_bound:.4e}"
    assert y_pred.shape == y_test.shape


@pytest.mark.slow
def test_performance_regression_large_fit() -> None:
    """Basic check to ensure fitting a large reservoir doesn't suddenly become slow."""
    n_neurons = 2000
    n_samples = 1000

    res = reservoirs.RandomSparse(n_neurons=n_neurons, spectral_radius=0.9)
    readout = readouts.Ridge(alpha=1e-6, include_bias=True)
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    rng = np.random.default_rng(seed=42)
    x = rng.random((n_samples, 1))
    y = rng.random((n_samples, 1))

    start_time = time.perf_counter()
    model.fit(x, y)
    end_time = time.perf_counter()

    duration = end_time - start_time

    # 2000 neurons should fit within ~2 seconds on modern hardware.
    # This is a loose bound to catch massive regressions (e.g. O(N^3) instead of O(N^2)).
    max_fit_time = 5.0
    assert duration < max_fit_time, f"Large fit took {duration:.2f}s, which is slower than expected {max_fit_time}s"
