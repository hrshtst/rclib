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

    rng = np.random.default_rng(seed=42)
    x_train = rng.random((20, 1))
    y_train = rng.random((20, 1))
    model.fit(x_train, y_train)

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


def test_model_partial_fit() -> None:
    """Test partial_fit for online learning."""
    model = ESN()
    res = reservoirs.RandomSparse(n_neurons=100, spectral_radius=0.9)
    readout = readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True)
    model.add_reservoir(res)
    model.set_readout(readout)

    rng = np.random.default_rng(seed=42)
    x = rng.random((1, 1))
    y = rng.random((1, 1))

    # Initial fit to allocate weights
    model.partial_fit(x, y)
    pred_before = model.predict(x)

    # Further fit
    model.partial_fit(x, y)
    pred_after = model.predict(x)

    assert not np.allclose(pred_before, pred_after)


def test_parallel_model_partial_fit() -> None:
    """Test partial_fit with parallel connection."""
    model = ESN(connection_type="parallel")
    res1 = reservoirs.RandomSparse(n_neurons=50, spectral_radius=0.9)
    res2 = reservoirs.RandomSparse(n_neurons=50, spectral_radius=0.9)
    readout = readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True)
    model.add_reservoir(res1)
    model.add_reservoir(res2)
    model.set_readout(readout)

    rng = np.random.default_rng(seed=42)
    x = rng.random((1, 1))
    y = rng.random((1, 1))

    # Should not raise any error
    model.partial_fit(x, y)


def test_nvar_model_fit_predict() -> None:
    """Test batch fit/predict with lazily-sized NVAR states."""
    mse_threshold = 1e-6
    model = ESN()
    model.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=2))
    model.set_readout(readouts.Ridge(alpha=1e-6, include_bias=True))

    x = np.linspace(0, 1, 50).reshape(-1, 1)
    y = 0.5 * x + 0.25

    model.fit(x, y, washout_len=2)
    y_pred = model.predict(x)

    assert y_pred.shape == y.shape
    assert np.mean((y_pred[2:] - y[2:]) ** 2) < mse_threshold


def test_model_minibatch_partial_fit() -> None:
    """Test model-level mini-batch partial_fit advances one row at a time."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=20, spectral_radius=0.9, seed=42))
    model.set_readout(readouts.Rls(lambda_=1.0, delta=1.0, include_bias=True, solver="rank_k_update"))

    rng = np.random.default_rng(seed=42)
    x = rng.random((8, 1))
    y = rng.random((8, 1))

    model.partial_fit(x, y)
    pred = model.predict(x)

    assert pred.shape == y.shape


def test_serial_model_partial_fit_none_uses_current_final_state() -> None:
    """Test x=None online update for serial multi-reservoir models."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9, seed=42))
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=5, spectral_radius=0.9, seed=43))
    model.set_readout(readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True))

    rng = np.random.default_rng(seed=42)
    x_train = rng.random((20, 1))
    y_train = rng.random((20, 1))
    model.fit(x_train, y_train)

    x = rng.random((1, 1))
    y = rng.random((1, 1))
    model.predict_online(x)
    model.partial_fit(None, y)


def test_parallel_model_partial_fit_none_uses_combined_current_state() -> None:
    """Test x=None online update for parallel multi-reservoir models."""
    model = ESN(connection_type="parallel")
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9, seed=42))
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=5, spectral_radius=0.9, seed=43))
    model.set_readout(readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True))

    rng = np.random.default_rng(seed=43)
    x_train = rng.random((20, 1))
    y_train = rng.random((20, 1))
    model.fit(x_train, y_train)

    x = rng.random((1, 1))
    y = rng.random((1, 1))
    model.predict_online(x)
    model.partial_fit(None, y)
