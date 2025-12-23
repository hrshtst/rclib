"""Tests for bias effect."""

from __future__ import annotations

import numpy as np
import pytest
from rclib.model import ESN
from rclib.readouts import Ridge
from rclib.reservoirs import RandomSparse


def _run_esn_experiment(*, include_bias: bool, input_scaling: float = 1.0) -> float:
    # --- Configuration Parameters ---
    n_total_samples = 200  # Smaller for faster tests
    n_train_samples = 150
    noise_amplitude = 0.01  # Reduced noise for clearer bias effect

    n_neurons = 100  # Smaller for faster tests
    spectral_radius = 0.9
    sparsity = 0.1
    leak_rate = 0.3
    ridge_alpha = 1e-6

    # --- Data Generation ---
    rng = np.random.default_rng(seed=42)
    time_np = np.linspace(0, 20, n_total_samples)
    clean_data = np.sin(time_np)
    noise = noise_amplitude * rng.standard_normal(n_total_samples)
    data = (clean_data + noise).reshape(-1, 1).astype(np.float64)

    input_data, target_data = data[:-1], data[1:]
    train_input = input_data[:n_train_samples]
    train_target = target_data[:n_train_samples]
    test_input = data[n_train_samples:-1]
    test_target = data[n_train_samples + 1 :]

    # --- Instantiate, Train, and Predict ---
    reservoir = RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=include_bias,
    )

    readout = Ridge(alpha=ridge_alpha, include_bias=include_bias)

    model = ESN(connection_type="serial")
    model.add_reservoir(reservoir)
    model.set_readout(readout)

    model.fit(train_input, train_target)
    predictions = model.predict(test_input, reset_state_before_predict=True)

    mse = np.mean((predictions[: len(test_target)] - test_target) ** 2)
    return float(mse)


def test_bias_effect_on_performance() -> None:
    """Test that bias parameter affects performance."""
    mse_with_bias = _run_esn_experiment(include_bias=True)
    mse_without_bias = _run_esn_experiment(include_bias=False)

    print(f"\nTest Bias Effect: MSE with bias={mse_with_bias:.6f}, MSE without bias={mse_without_bias:.6f}")

    # Assert that the MSE values are different, indicating that bias has an effect.
    assert mse_with_bias != pytest.approx(mse_without_bias, abs=1e-5)
