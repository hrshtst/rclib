# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for Reservoir classes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rclib import _rclib, readouts, reservoirs  # Import the C++ bindings
from rclib.model import ESN


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


def test_nvar_reservoir_polynomial_features() -> None:
    """Test NVAR polynomial feature ordering."""
    expected_output_dim = 14
    res = _rclib.NvarReservoir(num_lags=2, polynomial_order=2)

    res.advance(np.array([[1.0, 2.0]]))
    expected1 = np.array([[1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(res.getState(), expected1)

    res.advance(np.array([[3.0, 4.0]]))
    expected2 = np.array([[3.0, 4.0, 1.0, 2.0, 9.0, 12.0, 3.0, 6.0, 16.0, 4.0, 8.0, 1.0, 2.0, 4.0]])
    assert np.allclose(res.getState(), expected2)
    assert res.getOutputDim(2) == expected_output_dim


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


def test_reservoir_validation() -> None:
    """Test public reservoir config validation."""
    from rclib import reservoirs

    with np.testing.assert_raises(ValueError):
        reservoirs.RandomSparse(n_neurons=0, spectral_radius=0.9)
    with np.testing.assert_raises(ValueError):
        reservoirs.RandomSparse(n_neurons=10, spectral_radius=-0.1)
    with np.testing.assert_raises(ValueError):
        reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9, sparsity=1.5)
    with np.testing.assert_raises(ValueError):
        reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9, leak_rate=0.0)
    with np.testing.assert_raises(ValueError):
        reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9, input_scaling=-1.0)
    with np.testing.assert_raises(ValueError):
        reservoirs.Nvar(num_lags=0)
    with np.testing.assert_raises(ValueError):
        reservoirs.Nvar(num_lags=1, polynomial_order=0)
    with np.testing.assert_raises(ValueError):
        reservoirs.Nvar(num_lags=1, polynomial_order=33)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["spectral_radius", "sparsity", "leak_rate", "input_scaling"])
def test_random_sparse_rejects_non_finite(name: str, bad: float) -> None:
    """Non-finite hyperparameters are rejected by the config class and by the C++ constructor."""
    params = {"spectral_radius": 0.9, "sparsity": 0.5, "leak_rate": 0.5, "input_scaling": 1.0} | {name: bad}
    args = (params["spectral_radius"], params["sparsity"], params["leak_rate"], params["input_scaling"])
    with pytest.raises(ValueError, match=name):
        reservoirs.RandomSparse(10, *args)
    with pytest.raises(ValueError, match=name):
        _rclib.RandomSparseReservoir(10, *args)


def _online_model() -> tuple[ESN, np.ndarray, np.ndarray]:
    """A one-reservoir model that has already learned one sample online, with that sample."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9))
    model.set_readout(readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True))
    rng = np.random.default_rng(seed=42)
    x = rng.random((1, 1))
    y = rng.random((1, 1))
    model.partial_fit(x, y)
    return model, x, y


def _assert_one_reservoir_and_still_learning(model: ESN, x: np.ndarray, y: np.ndarray) -> None:
    assert len(model._reservoirs_params) == 1  # noqa: SLF001
    assert model._cpp_model.getNumReservoirs() == 1  # noqa: SLF001
    model.partial_fit(None, y)  # learns from the current reservoir state
    model.partial_fit(x, y)


def test_random_sparse_rejects_unknown_spectral_radius_method() -> None:
    """Unknown spectral_radius_method names are rejected by the config class and by ESN.add_reservoir.

    A rejected reservoir leaves the model unchanged, so it keeps learning.
    """
    with pytest.raises(ValueError, match="spectral_radius_method"):
        reservoirs.RandomSparse(10, 0.9, spectral_radius_method="arnoldi")

    model, x, y = _online_model()
    config = reservoirs.RandomSparse(10, 0.9)
    config.spectral_radius_method = "arnoldi"
    with pytest.raises(ValueError, match="spectral_radius_method"):
        model.add_reservoir(config)

    _assert_one_reservoir_and_still_learning(model, x, y)


@pytest.mark.parametrize("bias_scaling", [-0.1, float("nan"), float("inf")])
def test_random_sparse_rejects_an_invalid_bias_scaling(bias_scaling: float) -> None:
    """A negative or non-finite bias_scaling is rejected by the config class and by the C++ reservoir."""
    with pytest.raises(ValueError, match="bias_scaling"):
        reservoirs.RandomSparse(10, 0.9, include_bias=True, bias_scaling=bias_scaling)
    with pytest.raises(ValueError, match="bias_scaling"):
        _rclib.RandomSparseReservoir(10, 0.9, 0.5, 0.5, 1.0, include_bias=True, bias_scaling=bias_scaling)


def test_bias_scaling_scales_the_bias_alone() -> None:
    """The bias is the unscaled draw times bias_scaling; the weights are the same at every bias_scaling."""

    def build(bias_scaling: float, *, include_bias: bool = True) -> _rclib.RandomSparseReservoir:
        res = _rclib.RandomSparseReservoir(
            20, 0.9, 0.3, 0.5, 2.0, include_bias=include_bias, seed=7, bias_scaling=bias_scaling
        )
        res.advance(np.ones((1, 3)))  # draws W_in
        return res

    scaling = 0.3
    unscaled, scaled, zero = build(1.0), build(scaling), build(0.0)
    by_default = _rclib.RandomSparseReservoir(20, 0.9, 0.3, 0.5, 2.0, include_bias=True, seed=7)
    np.testing.assert_array_equal(by_default.getBias(), unscaled.getBias())
    assert 0 < np.abs(unscaled.getBias()).max() <= 1

    np.testing.assert_array_equal(scaled.getBias(), unscaled.getBias() * scaling)
    assert not np.any(zero.getBias())
    for res in (scaled, zero):
        for got, want in zip(res.getRecurrentWeights(), unscaled.getRecurrentWeights(), strict=True):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(res.getInputWeights(), unscaled.getInputWeights())
    assert not np.any(build(scaling, include_bias=False).getBias())

    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(20, 0.9, 0.3, 0.5, 2.0, include_bias=True, seed=7, bias_scaling=scaling))
    assert esn.get_reservoir(0).getBiasScaling() == scaling
    np.testing.assert_array_equal(esn.get_reservoir(0).getBias(), scaled.getBias())


def test_unsupported_reservoir_type_leaves_the_model_unchanged() -> None:
    """ESN.add_reservoir rejects an unknown reservoir type without recording it."""
    model, x, y = _online_model()
    with pytest.raises(TypeError, match="Unsupported reservoir type"):
        model.add_reservoir(object())

    _assert_one_reservoir_and_still_learning(model, x, y)


def test_reservoir_getters_and_input_dim() -> None:
    """Configuration getters return constructor values; getInputDim follows the first input."""
    methods = _rclib.RandomSparseReservoir.SpectralRadiusMethod
    res = _rclib.RandomSparseReservoir(10, 0.9, 0.5, 0.25, 2.0, include_bias=True, seed=7)
    assert (res.getNNeurons(), res.getSpectralRadius(), res.getSparsity()) == (10, 0.9, 0.5)
    assert (res.getLeakRate(), res.getInputScaling(), res.getIncludeBias(), res.getSeed()) == (0.25, 2.0, True, 7)
    assert res.getSpectralRadiusMethod() == methods.POWER_ITERATION
    scaled = _rclib.RandomSparseReservoir(10, 0.9, 0.5, 0.25, 2.0, bias_scaling=0.5)
    assert (res.getBiasScaling(), scaled.getBiasScaling()) == (1.0, 0.5)
    dense = _rclib.RandomSparseReservoir(10, 0.9, 0.5, 0.25, 2.0, spectral_radius_method=methods.DENSE)
    assert dense.getSpectralRadiusMethod() == methods.DENSE
    assert res.getInputDim() == 0
    x = np.ones((1, 3))
    res.advance(x)
    assert res.getInputDim() == x.shape[1]

    nvar = _rclib.NvarReservoir(num_lags=3, polynomial_order=2)
    assert (nvar.getNumLags(), nvar.getPolynomialOrder(), nvar.getInputDim()) == (3, 2, 0)
    x = np.ones((1, 2))
    nvar.advance(x)
    assert nvar.getInputDim() == x.shape[1]


def test_reservoir_weights_are_readable_copies() -> None:
    """The weights rebuild advance() exactly; they are copies, and W_in exists only after the first advance."""
    leak_rate = 0.25
    methods = _rclib.RandomSparseReservoir.SpectralRadiusMethod
    res = _rclib.RandomSparseReservoir(
        20, 0.9, 0.3, leak_rate, 2.0, include_bias=True, seed=7, spectral_radius_method=methods.DENSE
    )
    with pytest.raises(RuntimeError, match="advance before getInputWeights"):
        res.getInputWeights()

    rows, cols, values = res.getRecurrentWeights()
    w_res = np.zeros((20, 20))
    w_res[rows, cols] = values
    assert len(values) == np.count_nonzero(w_res)  # one entry per nonzero
    assert np.isclose(np.abs(np.linalg.eigvals(w_res)).max(), 0.9, rtol=1e-10, atol=0.0)
    bias = res.getBias()
    assert bias.shape == (20,)
    assert np.any(bias != 0)
    assert not np.any(_rclib.RandomSparseReservoir(20, 0.9, 0.3, leak_rate, 2.0, seed=7).getBias())

    rng = np.random.default_rng(seed=7)
    state = np.zeros((1, 20))
    for _ in range(3):
        u = rng.uniform(-1.0, 1.0, (1, 3))
        res.advance(u)
        w_in = res.getInputWeights()
        assert w_in.shape == (3, 20)
        state = (1 - leak_rate) * state + leak_rate * np.tanh(u @ w_in + bias + state @ w_res)
        assert np.allclose(res.getState(), state, atol=1e-12, rtol=0.0)

    w_in[:] = 0.0  # changing the copies leaves the reservoir as it was
    values[:] = 0.0
    bias[:] = 0.0
    assert np.any(res.getInputWeights() != 0)
    assert np.any(res.getRecurrentWeights()[2] != 0)
    assert np.any(res.getBias() != 0)
