"""Tests for the Model class."""

from __future__ import annotations

import numpy as np
from rclib import readouts, reservoirs
from rclib.model import ESN


def test_model_creation() -> None:
    """Test model creation."""
    model = ESN()
    assert model is not None


def test_model_fit_predict() -> None:
    """Test model fitting and prediction."""
    model = ESN()
    res = reservoirs.RandomSparse(
        n_neurons=100, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2, include_bias=False, input_scaling=1.0
    )
    readout = readouts.Ridge(alpha=1e-6, include_bias=False)
    model.add_reservoir(res)
    model.set_readout(readout)

    rng = np.random.default_rng(seed=42)
    x_train = rng.random((200, 1))
    y_train = rng.random((200, 1))

    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)

    assert y_pred.shape == (200, 1)
    assert np.mean((y_pred - y_train) ** 2) < np.mean(y_train**2)


def test_parallel_model_fit_predict() -> None:
    """Test parallel model fitting and prediction."""
    model = ESN(connection_type="parallel")
    res1 = reservoirs.RandomSparse(
        n_neurons=50, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2, include_bias=False, input_scaling=1.0
    )
    res2 = reservoirs.RandomSparse(
        n_neurons=50, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2, include_bias=False, input_scaling=1.0
    )
    readout = readouts.Ridge(alpha=1e-6, include_bias=False)
    model.add_reservoir(res1)
    model.add_reservoir(res2)
    model.set_readout(readout)

    rng = np.random.default_rng(seed=42)
    x_train = rng.random((200, 1))
    y_train = rng.random((200, 1))

    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)

    assert y_pred.shape == (200, 1)
    assert np.mean((y_pred - y_train) ** 2) < np.mean(y_train**2)


def test_model_reset_reservoirs() -> None:
    """Test reservoir reset."""
    model = ESN()
    res1 = reservoirs.RandomSparse(
        n_neurons=10, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2, include_bias=False, input_scaling=1.0
    )
    res2 = reservoirs.RandomSparse(
        n_neurons=5, spectral_radius=0.8, sparsity=0.2, leak_rate=0.3, include_bias=False, input_scaling=1.0
    )
    model.add_reservoir(res1)
    model.add_reservoir(res2)
    readout = readouts.Ridge(alpha=1e-6, include_bias=False)
    model.set_readout(readout)

    # Advance states to ensure they are not zero
    input_data = np.ones((10, 1))
    model.predict(input_data, reset_state_before_predict=False)

    # Check that states are not zero
    assert np.linalg.norm(model.get_reservoir(0).getState()) > 0
    assert np.linalg.norm(model.get_reservoir(1).getState()) > 0

    model.reset_reservoirs()

    # Check that states are reset to zero
    assert np.linalg.norm(model.get_reservoir(0).getState()) == 0
    assert np.linalg.norm(model.get_reservoir(1).getState()) == 0
