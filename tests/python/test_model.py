# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Model class."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from rclib import _rclib, readouts, reservoirs
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


@pytest.mark.parametrize(
    "reservoir",
    [reservoirs.Nvar(num_lags=2), reservoirs.RandomSparse(n_neurons=10, spectral_radius=0.9)],
    ids=["nvar", "random_sparse"],
)
def test_parallel_model_reservoir_error_raises(reservoir: reservoirs.Nvar | reservoirs.RandomSparse) -> None:
    """Test that a reservoir error inside the parallel update raises instead of aborting."""
    model = ESN(connection_type="parallel")
    # Each add_reservoir call creates a separate C++ reservoir from the configuration.
    model.add_reservoir(reservoir)
    model.add_reservoir(reservoir)
    model.set_readout(readouts.Ridge(alpha=1e-6, include_bias=True))

    rng = np.random.default_rng(seed=42)
    x = rng.random((20, 1))
    model.fit(x, x)

    # Reservoirs lock their input width on first use, so a wider input must be rejected.
    with pytest.raises(ValueError, match="input dimension changed"):
        model.predict(rng.random((5, 2)))

    # The model remains usable with the original input width.
    assert model.predict(x).shape == x.shape


def test_model_reservoir_count_and_connection_type() -> None:
    """The C++ model reports its reservoir count and connection type."""
    model = ESN(connection_type="parallel")
    configs = [reservoirs.Nvar(num_lags=2), reservoirs.Nvar(num_lags=3)]
    for config in configs:
        model.add_reservoir(config)
    assert model._cpp_model.getNumReservoirs() == len(configs)  # noqa: SLF001
    assert model._cpp_model.getConnectionType() == "parallel"  # noqa: SLF001


def _generative_model(output_columns: int = 1) -> ESN:
    """A model trained to predict the next value of a sine wave."""
    model = ESN(connection_type="parallel")
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=30, spectral_radius=0.9, include_bias=True, seed=7))
    model.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=2))
    model.set_readout(readouts.Ridge(alpha=1e-4, include_bias=True))
    series = np.sin(0.3 * np.arange(81)).reshape(-1, 1)
    model.fit(series[:-1], np.repeat(series[1:], output_columns, axis=1), washout_len=10)
    return model


@pytest.mark.parametrize("chunks", [[3, 4], [6, 1], [1] * 7])
def test_predict_generative_in_chunks_equals_one_call(chunks: list[int]) -> None:
    """Generating in chunks continues the sequence and ends in the same reservoir states."""
    trained = _generative_model()
    prime = np.sin(0.3 * np.arange(10) + 2.0).reshape(-1, 1)
    whole = copy.deepcopy(trained)
    expected = whole.predict_generative(prime, sum(chunks))

    chunked = copy.deepcopy(trained)
    parts = [chunked.predict_generative(prime if i == 0 else np.empty((0, 1)), n) for i, n in enumerate(chunks)]
    np.testing.assert_array_equal(np.vstack(parts), expected)
    for i in range(2):
        np.testing.assert_array_equal(chunked.get_reservoir(i).getState(), whole.get_reservoir(i).getState())


def test_predict_generative_zero_steps_feeds_nothing_back() -> None:
    """Zero steps consume the priming data but feed no output back."""
    trained = _generative_model()
    prime = np.sin(0.3 * np.arange(10) + 2.0).reshape(-1, 1)
    generative = copy.deepcopy(trained)
    online = copy.deepcopy(trained)
    assert generative.predict_generative(prime, 0).shape == (0, 1)
    online.predict_online(prime)
    for i in range(2):
        np.testing.assert_array_equal(generative.get_reservoir(i).getState(), online.get_reservoir(i).getState())


@pytest.mark.parametrize("n_steps", [1, 3])
def test_predict_generative_rejects_outputs_that_cannot_be_fed_back(n_steps: int) -> None:
    """Outputs wider than the model input raise ValueError, also for a single step."""
    model = _generative_model(output_columns=2)
    prime = np.sin(0.3 * np.arange(10)).reshape(-1, 1)
    with pytest.raises(ValueError, match="input dimension changed"):
        model.predict_generative(prime, n_steps)


def _sequence_model() -> ESN:
    """An untrained serial RandomSparse -> NVAR model; every call builds identical reservoirs."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=8, spectral_radius=0.9, include_bias=True, seed=7))
    model.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=2))
    model.set_readout(readouts.Ridge(alpha=1e-4, include_bias=True))
    return model


def _sine(n_samples: int, phase: float) -> np.ndarray:
    return np.sin(0.3 * np.arange(n_samples) + phase).reshape(-1, 1)


def _episodes(phase_offset: float) -> list[np.ndarray]:
    return [_sine(20, phase_offset), _sine(25, 1.0 + phase_offset), _sine(15, 2.0 + phase_offset)]


def _reference_states(model: ESN, x: np.ndarray) -> np.ndarray:
    """States of a serial model for x from reset reservoirs, collected with the reservoir API."""
    states = x
    for i in range(model._cpp_model.getNumReservoirs()):  # noqa: SLF001
        reservoir = model.get_reservoir(i)
        reservoir.resetState()
        states = np.vstack([reservoir.advance(row.reshape(1, -1)) for row in states])
    return states


@pytest.mark.parametrize("washout_len", [0, 5])
def test_fit_sequences_with_one_sequence_equals_fit(washout_len: int) -> None:
    """A single sequence gives bit-identical predictions to fit."""
    x, y = _sine(40, 0.0), _sine(40, 0.3)
    single = _sequence_model()
    single.fit(x, y, washout_len=washout_len)
    sequences = _sequence_model()
    sequences.fit_sequences([x], [y], washout_len=washout_len)

    probe = _sine(10, 2.0)
    np.testing.assert_array_equal(sequences.predict(probe), single.predict(probe))


@pytest.mark.parametrize("washout_len", [0, 3])
def test_fit_sequences_fits_one_readout_on_per_sequence_states(washout_len: int) -> None:
    """The readout equals one fitted on the stacked states of each sequence from reset reservoirs."""
    inputs, targets = _episodes(0.0), _episodes(0.3)
    model = _sequence_model()
    model.fit_sequences(inputs, targets, washout_len=washout_len)

    states = np.vstack([_reference_states(model, x)[washout_len:] for x in inputs])
    reference = _rclib.RidgeReadout(1e-4, True, _rclib.RidgeReadout.Solver.AUTO, 1e-10)  # noqa: FBT003
    reference.fit(states, np.vstack([y[washout_len:] for y in targets]))
    probe = _sine(10, 2.0)
    expected = reference.predict(_reference_states(model, probe))
    np.testing.assert_array_equal(model.predict(probe), expected)


def test_fit_sequences_learns_nothing_across_sequence_boundaries() -> None:
    """Reordering the sequences changes the predictions only by floating-point summation order."""
    inputs, targets = _episodes(0.0), _episodes(0.3)
    forward = _sequence_model()
    forward.fit_sequences(inputs, targets, washout_len=3)
    backward = _sequence_model()
    backward.fit_sequences(inputs[::-1], targets[::-1], washout_len=3)

    probe = _sine(30, 0.7)
    np.testing.assert_allclose(backward.predict(probe), forward.predict(probe), rtol=1e-9, atol=1e-12)


def test_fit_sequences_accepts_tuples() -> None:
    """A tuple of arrays works like a list."""
    inputs, targets = _episodes(0.0), _episodes(0.3)
    from_list, from_tuple = _sequence_model(), _sequence_model()
    from_list.fit_sequences(inputs, targets, washout_len=2)
    from_tuple.fit_sequences(tuple(inputs), tuple(targets), washout_len=2)

    probe = _sine(10, 2.0)
    np.testing.assert_array_equal(from_tuple.predict(probe), from_list.predict(probe))


@pytest.mark.parametrize(
    ("inputs", "targets", "washout_len", "error", "match"),
    [
        ([], [], 0, ValueError, "inputs must hold at least one sequence"),
        ([_sine(10, 0.0)], [_sine(10, 0.3), _sine(12, 1.3)], 0, ValueError, "as many sequences as inputs"),
        ([_sine(10, 0.0), np.empty((0, 1))], [_sine(10, 0.3), np.empty((0, 1))], 0, ValueError, "sequence 1: inputs"),
        ([_sine(10, 0.0), _sine(12, 1.0)], [_sine(10, 0.3), _sine(11, 1.3)], 0, ValueError, "sequence 1: targets"),
        ([_sine(10, 0.0), _sine(12, 1.0)], [_sine(10, 0.3), np.empty((12, 0))], 0, ValueError, "sequence 1: targets"),
        ([_sine(10, 0.0)], [_sine(10, 0.3)], -1, IndexError, "washout_len must be non-negative"),
        ([_sine(10, 0.0), _sine(2, 1.0)], [_sine(10, 0.3), _sine(2, 1.3)], 2, IndexError, "sequence 1: washout_len"),
        ([_sine(10, 0.0), np.ones((12, 2))], [_sine(10, 0.3), _sine(12, 1.3)], 0, ValueError, "sequence 1: inputs"),
        ([_sine(10, 0.0), _sine(12, 1.0)], [_sine(10, 0.3), np.ones((12, 2))], 0, ValueError, "sequence 1: targets"),
    ],
)
def test_fit_sequences_rejects_invalid_sequences(
    inputs: list[np.ndarray],
    targets: list[np.ndarray],
    washout_len: int,
    error: type[Exception],
    match: str,
) -> None:
    """Invalid sequences raise before anything changes, naming the sequence where it applies."""
    model, untouched = _sequence_model(), _sequence_model()
    for fitted in (model, untouched):
        fitted.fit(_sine(20, 0.0), _sine(20, 0.3))

    with pytest.raises(error, match=match):
        model.fit_sequences(inputs, targets, washout_len=washout_len)
    for i in range(2):
        np.testing.assert_array_equal(model.get_reservoir(i).getState(), untouched.get_reservoir(i).getState())
    probe = _sine(10, 2.0)
    np.testing.assert_array_equal(model.predict(probe), untouched.predict(probe))


def test_fit_sequences_rejects_a_single_array() -> None:
    """A single 2-D array, which would split into one-row sequences, is rejected."""
    model = _sequence_model()
    x, y = _sine(10, 0.0), _sine(10, 0.3)
    # Type errors on purpose: these are the mistakes the check catches at run time.
    with pytest.raises(ValueError, match="inputs must be a sequence of 2-D arrays"):
        model.fit_sequences(x, [y])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="targets must be a sequence of 2-D arrays"):
        model.fit_sequences([x], y)  # pyright: ignore[reportArgumentType]


def test_fit_sequences_requires_a_configured_model() -> None:
    """A model without a readout cannot be fitted."""
    model = ESN()
    model.add_reservoir(reservoirs.Nvar(num_lags=2))
    with pytest.raises(RuntimeError, match="not fully configured"):
        model.fit_sequences([_sine(10, 0.0)], [_sine(10, 0.3)])


def _width_model(topology: str) -> ESN:
    """An untrained RandomSparse, NVAR, serial RandomSparse -> NVAR or parallel RandomSparse + NVAR model."""
    model = ESN(connection_type="parallel" if topology == "parallel" else "serial")
    if topology in {"random_sparse", "serial", "parallel"}:
        model.add_reservoir(reservoirs.RandomSparse(n_neurons=8, spectral_radius=0.9, include_bias=True, seed=7))
    if topology in {"nvar", "serial", "parallel"}:
        model.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=2))
    model.set_readout(readouts.Ridge(alpha=1e-4, include_bias=True))
    return model


@pytest.mark.parametrize("method", ["fit", "fit_sequences"])
@pytest.mark.parametrize("topology", ["random_sparse", "nvar", "serial", "parallel"])
def test_retraining_with_another_input_width_changes_nothing(topology: str, method: str) -> None:
    """Input wider than the reservoirs are locked to is rejected before any reservoir is reset."""
    model = _width_model(topology)
    model.fit(_sine(20, 0.0), _sine(20, 0.3))
    model.predict_online(_sine(3, 1.0))  # states away from the reset ones
    twin = copy.deepcopy(model)
    n_reservoirs = model._cpp_model.getNumReservoirs()  # noqa: SLF001
    states = [model.get_reservoir(i).getState() for i in range(n_reservoirs)]

    wider, targets = np.ones((10, 2)), _sine(10, 0.3)
    if method == "fit":
        with pytest.raises(ValueError, match=r"^inputs have 2 columns, but the reservoirs expect 1\.$"):
            model.fit(wider, targets)
    else:
        with pytest.raises(ValueError, match=r"^sequence 0: inputs have 2 columns, but the reservoirs expect 1\.$"):
            model.fit_sequences([wider], [targets])

    for i in range(n_reservoirs):
        np.testing.assert_array_equal(model.get_reservoir(i).getState(), states[i])
    next_input = _sine(4, 1.0)[-1:]
    np.testing.assert_array_equal(model.predict_online(next_input), twin.predict_online(next_input))
    probe = _sine(10, 2.0)  # predict resets first, so this compares the readouts
    np.testing.assert_array_equal(model.predict(probe), twin.predict(probe))
