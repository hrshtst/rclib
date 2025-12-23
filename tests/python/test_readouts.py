"""Tests for Readout classes."""

from __future__ import annotations

import numpy as np
import pytest
from rclib import _rclib  # Import the C++ bindings


def test_ridge_readout_fit_predict() -> None:
    """Test Ridge Readout fitting and prediction."""
    n_samples = 100
    n_features = 10
    n_targets = 2

    rng = np.random.default_rng(seed=42)
    states = rng.random((n_samples, n_features))
    targets = rng.random((n_samples, n_targets))

    # Without bias
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=False)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=True)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_ridge_readout_partial_fit_error() -> None:
    """Test Ridge Readout error on partial fit."""
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=False)
    rng = np.random.default_rng(seed=42)
    state = rng.random((1, 10))
    target = rng.random((1, 2))

    with pytest.raises(RuntimeError):  # Expecting a RuntimeError from C++ for Ridge's partialFit
        readout.partialFit(state, target)


def test_lms_readout_fit_predict() -> None:
    """Test LMS Readout fitting and prediction."""
    n_samples = 100
    n_features = 10
    n_targets = 2

    rng = np.random.default_rng(seed=42)
    states = rng.random((n_samples, n_features))
    targets = rng.random((n_samples, n_targets))

    # Without bias
    readout = _rclib.LmsReadout(learning_rate=0.01, include_bias=False)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.LmsReadout(learning_rate=0.01, include_bias=True)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_lms_readout_partial_fit() -> None:
    """Test LMS Readout partial fit."""
    n_features = 5
    n_targets = 1
    readout = _rclib.LmsReadout(learning_rate=0.01, include_bias=False)

    rng = np.random.default_rng(seed=42)
    state1 = rng.random((1, n_features))
    target1 = rng.random((1, n_targets))

    readout.partialFit(state1, target1)
    predictions1 = readout.predict(state1)
    assert predictions1.shape == (1, n_targets)

    state2 = rng.random((1, n_features))
    target2 = rng.random((1, n_targets))
    readout.partialFit(state2, target2)
    predictions2 = readout.predict(state2)
    assert predictions2.shape == (1, n_targets)


def test_rls_readout_fit_predict() -> None:
    """Test RLS Readout fitting and prediction."""
    n_samples = 100
    n_features = 10
    n_targets = 2

    rng = np.random.default_rng(seed=42)
    states = rng.random((n_samples, n_features))
    targets = rng.random((n_samples, n_targets))

    # Without bias
    readout = _rclib.RlsReadout(lambda_=0.99, delta=1.0, include_bias=False)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.RlsReadout(lambda_=0.99, delta=1.0, include_bias=True)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_rls_readout_partial_fit() -> None:
    """Test RLS Readout partial fit."""
    n_features = 5
    n_targets = 1
    readout = _rclib.RlsReadout(lambda_=0.99, delta=1.0, include_bias=False)

    rng = np.random.default_rng(seed=42)
    state1 = rng.random((1, n_features))
    target1 = rng.random((1, n_targets))

    readout.partialFit(state1, target1)
    predictions1 = readout.predict(state1)
    assert predictions1.shape == (1, n_targets)

    state2 = rng.random((1, n_features))
    target2 = rng.random((1, n_targets))
    readout.partialFit(state2, target2)
    predictions2 = readout.predict(state2)
    assert predictions2.shape == (1, n_targets)
