"""Tests for Readout classes."""

from __future__ import annotations

import numpy as np
import pytest
from rclib import ESN, _rclib, readouts, reservoirs


def test_ridge_readout_fit_predict() -> None:
    """Test Ridge Readout fitting and prediction."""
    n_samples = 100
    n_features = 10
    n_targets = 2

    rng = np.random.default_rng(seed=42)
    states = rng.random((n_samples, n_features))
    targets = rng.random((n_samples, n_targets))

    # Without bias - CHOLESKY
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=False, solver=_rclib.RidgeReadout.Solver.CHOLESKY)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error

    # With bias - DUAL_CHOLESKY
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=True, solver=_rclib.RidgeReadout.Solver.DUAL_CHOLESKY)
    readout.fit(states, targets)
    predictions = readout.predict(states)

    assert predictions.shape == (n_samples, n_targets)
    prediction_error = np.linalg.norm(predictions - targets) ** 2
    original_error = np.linalg.norm(targets) ** 2
    assert prediction_error < original_error


def test_ridge_solver_consistency() -> None:
    """Test consistency between Ridge solvers."""
    n_samples = 50
    n_features = 80
    rng = np.random.default_rng(seed=42)
    states = rng.random((n_samples, n_features))
    targets = rng.random((n_samples, 1))

    alpha = 0.1
    # Primal
    primal = _rclib.RidgeReadout(alpha=alpha, include_bias=True, solver=_rclib.RidgeReadout.Solver.CHOLESKY)
    primal.fit(states, targets)
    pred_primal = primal.predict(states)

    # Dual
    dual = _rclib.RidgeReadout(alpha=alpha, include_bias=True, solver=_rclib.RidgeReadout.Solver.DUAL_CHOLESKY)
    dual.fit(states, targets)
    pred_dual = dual.predict(states)

    # Implicit
    implicit = _rclib.RidgeReadout(
        alpha=alpha, include_bias=True, solver=_rclib.RidgeReadout.Solver.CONJUGATE_GRADIENT_IMPLICIT
    )
    implicit.fit(states, targets)
    pred_implicit = implicit.predict(states)

    assert np.allclose(pred_primal, pred_dual, atol=1e-6)
    assert np.allclose(pred_primal, pred_implicit, atol=1e-6)


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


def test_rls_readout_solvers() -> None:
    """Test RLS Readout with different solver options."""
    n_features = 10
    n_targets = 1
    rng = np.random.default_rng(seed=42)
    states = rng.random((32, n_features))
    targets = rng.random((32, n_targets))

    # rank1_update
    rls1 = _rclib.RlsReadout(lambda_=1.0, delta=1.0, include_bias=True, solver=_rclib.RlsReadout.Solver.RANK1_UPDATE)
    rls1.partialFit(states, targets)
    pred1 = rls1.predict(states)

    # rank_k_update
    rlsk = _rclib.RlsReadout(lambda_=1.0, delta=1.0, include_bias=True, solver=_rclib.RlsReadout.Solver.RANK_K_UPDATE)
    rlsk.partialFit(states, targets)
    predk = rlsk.predict(states)

    assert np.allclose(pred1, predk, atol=1e-10)


def test_mini_batch_fit() -> None:
    """Test that partialFit handles mini-batches correctly."""
    n_features = 10
    n_targets = 1
    rng = np.random.default_rng(seed=42)
    states = rng.random((10, n_features))
    targets = rng.random((10, n_targets))

    # LMS
    lms = _rclib.LmsReadout(learning_rate=0.01, include_bias=True)
    lms.partialFit(states, targets)
    assert lms.predict(states).shape == (10, n_targets)

    # RLS
    rls = _rclib.RlsReadout(lambda_=0.99, delta=1.0, include_bias=True)
    rls.partialFit(states, targets)
    assert rls.predict(states).shape == (10, n_targets)


@pytest.mark.slow
def test_adaptive_solver_primal() -> None:
    """Test that a problem with N <= T uses the CHOLESKY solver."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=100, spectral_radius=0.9))
    model.set_readout(readouts.Ridge(alpha=1e-8, include_bias=True, solver="auto"))

    rng = np.random.default_rng(seed=42)
    x = rng.random((200, 1))
    y = rng.random((200, 1))
    model.fit(x, y)

    cpp_readout = model._cpp_model.getReadout()  # noqa: SLF001
    assert cpp_readout.getEffectiveSolver() == _rclib.RidgeReadout.Solver.CHOLESKY


@pytest.mark.slow
def test_adaptive_solver_dual() -> None:
    """Test that a problem with N > T uses the DUAL_CHOLESKY solver."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=1000, spectral_radius=0.9))
    model.set_readout(readouts.Ridge(alpha=1e-8, include_bias=True, solver="auto"))

    rng = np.random.default_rng(seed=42)
    x = rng.random((100, 1))
    y = rng.random((100, 1))
    model.fit(x, y)

    cpp_readout = model._cpp_model.getReadout()  # noqa: SLF001
    assert cpp_readout.getEffectiveSolver() == _rclib.RidgeReadout.Solver.DUAL_CHOLESKY


@pytest.mark.slow
def test_adaptive_solver_large() -> None:
    """Test that a large problem uses the CONJUGATE_GRADIENT_IMPLICIT solver."""
    model = ESN()
    # Total neurons >= 8000
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=8000, spectral_radius=0.9))
    model.set_readout(readouts.Ridge(alpha=1e-8, include_bias=True, solver="auto"))

    # Trigger fit to let C++ decide
    rng = np.random.default_rng(seed=42)
    x = rng.random((10, 1))
    y = rng.random((10, 1))
    model.fit(x, y)

    cpp_readout = model._cpp_model.getReadout()  # noqa: SLF001
    assert cpp_readout.getSolver() == _rclib.RidgeReadout.Solver.AUTO
    assert cpp_readout.getEffectiveSolver() == _rclib.RidgeReadout.Solver.CONJUGATE_GRADIENT_IMPLICIT


@pytest.mark.slow
def test_explicit_solver() -> None:
    """Test that an explicit solver choice overrides AUTO."""
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=10000, spectral_radius=0.9))
    model.set_readout(readouts.Ridge(alpha=1e-8, include_bias=True, solver="cholesky"))

    # Trigger fit
    rng = np.random.default_rng(seed=42)
    x = rng.random((10, 1))
    y = rng.random((10, 1))
    model.fit(x, y)

    cpp_readout = model._cpp_model.getReadout()  # noqa: SLF001
    assert cpp_readout.getSolver() == _rclib.RidgeReadout.Solver.CHOLESKY
    assert cpp_readout.getEffectiveSolver() == _rclib.RidgeReadout.Solver.CHOLESKY
