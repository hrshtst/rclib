"""Tests for Reservoir classes."""

from __future__ import annotations

import numpy as np
from rclib import _rclib  # Import the C++ bindings


def test_random_sparse_reservoir_init() -> None:
    """Test RandomSparseReservoir initialization."""
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=include_bias,
    )

    state = res.getState()
    assert state.shape == (1, n_neurons)
    assert np.all(state == 0)


def test_random_sparse_reservoir_advance() -> None:
    """Test RandomSparseReservoir advance."""
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=include_bias,
    )
    rng = np.random.default_rng(seed=42)
    input_data = rng.random((1, 5))

    for _ in range(10):
        res.advance(input_data)
        state = res.getState()
        assert state.shape == (1, n_neurons)
        assert not np.all(state == 0)


def test_random_sparse_reservoir_reset() -> None:
    """Test RandomSparseReservoir reset."""
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=include_bias,
    )
    rng = np.random.default_rng(seed=42)
    input_data = rng.random((1, 5))

    res.advance(input_data)
    assert not np.all(res.getState() == 0)

    res.resetState()
    assert np.all(res.getState() == 0)


def test_nvar_reservoir_init() -> None:
    """Test NvarReservoir initialization."""
    num_lags = 3
    res = _rclib.NvarReservoir(num_lags=num_lags)

    assert np.all(res.getState() == 0)


def test_nvar_reservoir_advance() -> None:
    """Test NvarReservoir advance."""
    num_lags = 3
    input_dim = 2
    res = _rclib.NvarReservoir(num_lags=num_lags)

    rng = np.random.default_rng(seed=42)
    input1 = rng.random((1, input_dim))
    input2 = rng.random((1, input_dim))
    input3 = rng.random((1, input_dim))

    res.advance(input1)
    state = res.getState()
    # Check first block (current input)
    assert np.allclose(state[:, :input_dim], input1)
    # Check other blocks (should be 0)
    assert np.all(state[:, input_dim:] == 0)

    res.advance(input2)
    state = res.getState()
    assert np.allclose(state[:, :input_dim], input2)
    assert np.allclose(state[:, input_dim : 2 * input_dim], input1)
    assert np.all(state[:, 2 * input_dim :] == 0)

    res.advance(input3)
    state = res.getState()
    assert np.allclose(state[:, :input_dim], input3)
    assert np.allclose(state[:, input_dim : 2 * input_dim], input2)
    assert np.allclose(state[:, 2 * input_dim : 3 * input_dim], input1)


def test_nvar_reservoir_reset() -> None:
    """Test NvarReservoir reset."""
    num_lags = 3
    input_dim = 2
    res = _rclib.NvarReservoir(num_lags=num_lags)
    rng = np.random.default_rng(seed=42)
    input_data = rng.random((1, input_dim))

    res.advance(input_data)
    assert not np.all(res.getState() == 0)

    res.resetState()
    assert np.all(res.getState() == 0)
