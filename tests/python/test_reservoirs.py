from __future__ import annotations

import numpy as np
from rclib import _rclib  # Import the C++ bindings


def test_random_sparse_reservoir_init():
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias)

    state = res.getState()
    assert state.shape == (1, n_neurons)
    assert np.all(state == 0)


def test_random_sparse_reservoir_advance():
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias)
    input_data = np.random.rand(1, 5)

    for _ in range(10):
        res.advance(input_data)
        state = res.getState()
        assert state.shape == (1, n_neurons)
        assert not np.all(state == 0)


def test_random_sparse_reservoir_reset():
    n_neurons = 10
    spectral_radius = 0.9
    sparsity = 0.5
    leak_rate = 0.1
    input_scaling = 1.0  # Added missing argument
    include_bias = True

    res = _rclib.RandomSparseReservoir(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias)
    input_data = np.random.rand(1, 5)

    res.advance(input_data)
    assert not np.all(res.getState() == 0)

    res.resetState()  # Corrected method name: resetState
    assert np.all(res.getState() == 0)


def test_nvar_reservoir_init():
    num_lags = 3
    res = _rclib.NvarReservoir(num_lags)  # Corrected class name: NvarReservoir

    assert np.all(res.getState() == 0)  # Corrected method name: getState


def test_nvar_reservoir_advance():
    num_lags = 3
    input_dim = 2
    res = _rclib.NvarReservoir(num_lags)  # Corrected class name: NvarReservoir

    input1 = np.random.rand(1, input_dim)
    input2 = np.random.rand(1, input_dim)
    input3 = np.random.rand(1, input_dim)

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


def test_nvar_reservoir_reset():
    num_lags = 3
    input_dim = 2
    res = _rclib.NvarReservoir(num_lags)  # Corrected class name: NvarReservoir
    input_data = np.random.rand(1, input_dim)

    res.advance(input_data)
    assert not np.all(res.getState() == 0)

    res.resetState()  # Corrected method name: resetState
    assert np.all(res.getState() == 0)
