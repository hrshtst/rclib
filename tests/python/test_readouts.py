from __future__ import annotations

import numpy as np
import pytest
from rclib import _rclib  # Import the C++ bindings


def test_ridge_readout_fit_predict():
    n_samples = 100
    n_features = 10
    n_targets = 2

    states = np.random.rand(n_samples, n_features)
    targets = np.random.rand(n_samples, n_targets)

    # Without bias
    readout = _rclib.RidgeReadout(0.1, False)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.RidgeReadout(0.1, True)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_ridge_readout_partial_fit_error():
    readout = _rclib.RidgeReadout(0.1, False)  # Provide alpha and include_bias
    state = np.random.rand(1, 10)
    target = np.random.rand(1, 2)

    with pytest.raises(RuntimeError):  # Expecting a RuntimeError from C++ for Ridge's partialFit
        readout.partialFit(state, target)


def test_lms_readout_fit_predict():
    n_samples = 100
    n_features = 10
    n_targets = 2

    states = np.random.rand(n_samples, n_features)
    targets = np.random.rand(n_samples, n_targets)

    # Without bias
    readout = _rclib.LmsReadout(0.01, False)  # Corrected case: LmsReadout
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.LmsReadout(0.01, True)  # Corrected case: LmsReadout
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_lms_readout_partial_fit():
    n_features = 5
    n_targets = 1
    readout = _rclib.LmsReadout(0.01, False)  # Corrected case: LmsReadout

    state1 = np.random.rand(1, n_features)
    target1 = np.random.rand(1, n_targets)

    readout.partialFit(state1, target1)  # Corrected method name: partialFit
    predictions1 = readout.predict(state1)
    assert predictions1.shape == (1, n_targets)

    state2 = np.random.rand(1, n_features)
    target2 = np.random.rand(1, n_targets)
    readout.partialFit(state2, target2)  # Corrected method name: partialFit
    predictions2 = readout.predict(state2)
    assert predictions2.shape == (1, n_targets)


def test_rls_readout_fit_predict():
    n_samples = 100
    n_features = 10
    n_targets = 2

    states = np.random.rand(n_samples, n_features)
    targets = np.random.rand(n_samples, n_targets)

    # Without bias
    readout = _rclib.RlsReadout(0.99, 1.0, False)  # Corrected case: RlsReadout
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias
    readout = _rclib.RlsReadout(0.99, 1.0, True)  # Corrected case: RlsReadout
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_rls_readout_partial_fit():
    n_features = 5
    n_targets = 1
    readout = _rclib.RlsReadout(0.99, 1.0, False)  # Corrected case: RlsReadout

    state1 = np.random.rand(1, n_features)
    target1 = np.random.rand(1, n_targets)

    readout.partialFit(state1, target1)  # Corrected method name: partialFit
    predictions1 = readout.predict(state1)
    assert predictions1.shape == (1, n_targets)

    state2 = np.random.rand(1, n_features)
    target2 = np.random.rand(1, n_targets)
    readout.partialFit(state2, target2)  # Corrected method name: partialFit
    predictions2 = readout.predict(state2)
    assert predictions2.shape == (1, n_targets)
