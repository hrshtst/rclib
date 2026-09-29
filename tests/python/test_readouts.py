# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for Readout classes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rclib import ESN, _rclib, readouts, reservoirs

if TYPE_CHECKING:
    from collections.abc import Callable


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


def test_readout_validation() -> None:
    """Test public readout config validation rejects invalid parameters."""
    with pytest.raises(ValueError, match="alpha"):
        readouts.Ridge(alpha=-1.0, include_bias=True)
    with pytest.raises(ValueError, match="tolerance"):
        readouts.Ridge(alpha=1.0, include_bias=True, tolerance=0.0)
    with pytest.raises(ValueError, match="lambda"):
        readouts.Rls(lambda_=1.5, delta=1.0, include_bias=True)
    with pytest.raises(ValueError, match="delta"):
        readouts.Rls(lambda_=0.99, delta=0.0, include_bias=True)
    with pytest.raises(ValueError, match="learning_rate"):
        readouts.Lms(learning_rate=0.0, include_bias=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("name", "make_config", "make_cpp"),
    [
        (
            "alpha",
            lambda v: readouts.Ridge(alpha=v, include_bias=True),
            lambda v: _rclib.RidgeReadout(alpha=v, include_bias=True),
        ),
        (
            "tolerance",
            lambda v: readouts.Ridge(alpha=1.0, include_bias=True, tolerance=v),
            lambda v: _rclib.RidgeReadout(alpha=1.0, include_bias=True, tolerance=v),
        ),
        (
            "lambda",
            lambda v: readouts.Rls(lambda_=v, delta=1.0, include_bias=True),
            lambda v: _rclib.RlsReadout(lambda_=v, delta=1.0, include_bias=True),
        ),
        (
            "delta",
            lambda v: readouts.Rls(lambda_=0.99, delta=v, include_bias=True),
            lambda v: _rclib.RlsReadout(lambda_=0.99, delta=v, include_bias=True),
        ),
        (
            "learning_rate",
            lambda v: readouts.Lms(learning_rate=v, include_bias=True),
            lambda v: _rclib.LmsReadout(learning_rate=v, include_bias=True),
        ),
    ],
)
def test_readout_rejects_non_finite(
    name: str, make_config: Callable[[float], object], make_cpp: Callable[[float], object], bad: float
) -> None:
    """Non-finite hyperparameters are rejected by the config classes and by the C++ constructors."""
    with pytest.raises(ValueError, match=name):
        make_config(bad)
    with pytest.raises(ValueError, match=name):
        make_cpp(bad)


@pytest.mark.parametrize(
    ("states_shape", "targets_shape"),
    [((0, 4), (0, 2)), ((5, 0), (5, 2)), ((5, 4), (3, 2)), ((5, 4), (7, 2)), ((5, 4), (5, 0))],
    ids=["no_rows", "no_state_columns", "fewer_target_rows", "more_target_rows", "no_target_columns"],
)
def test_lms_fit_rejects_invalid_input(states_shape: tuple[int, int], targets_shape: tuple[int, int]) -> None:
    """LMS fit rejects malformed input before resetting, keeping a trained readout intact."""
    rng = np.random.default_rng(seed=5)
    states = rng.random((5, 4))
    readout = _rclib.LmsReadout(learning_rate=0.05, include_bias=True)
    readout.fit(states, rng.random((5, 2)))
    before = readout.predict(states)

    with pytest.raises(ValueError, match=r"states|targets"):
        readout.fit(rng.random(states_shape), rng.random(targets_shape))
    np.testing.assert_array_equal(readout.predict(states), before)


def test_ridge_readout_weights_are_readable_and_read_only() -> None:
    """The fitted weights are exposed as a copy with the bias row last; reading them never alters prediction."""
    rng = np.random.default_rng(seed=7)
    states = rng.random((60, 8))
    targets = rng.random((60, 3))
    readout = _rclib.RidgeReadout(alpha=0.1, include_bias=True, solver=_rclib.RidgeReadout.Solver.CHOLESKY)
    with pytest.raises(RuntimeError, match="must be fit before getWeights"):
        readout.getWeights()
    assert readout.getIncludeBias() is True
    readout.fit(states, targets)
    weights = readout.getWeights()
    assert weights.shape == (9, 3)
    expected = states @ weights[:-1] + weights[-1]
    assert np.allclose(readout.predict(states), expected, atol=1e-12, rtol=0.0)
    weights[:] = 0.0  # a copy: mutating it leaves the readout untouched
    assert np.allclose(readout.predict(states), expected, atol=1e-12, rtol=0.0)
    plain = _rclib.RidgeReadout(alpha=0.1, include_bias=False, solver=_rclib.RidgeReadout.Solver.DUAL_CHOLESKY)
    plain.fit(states, targets)
    assert plain.getIncludeBias() is False
    assert plain.getWeights().shape == (8, 3)
    assert np.allclose(plain.predict(states), states @ plain.getWeights(), atol=1e-12, rtol=0.0)


def test_readout_getters_and_input_dim() -> None:
    """Configuration getters return constructor values; getInputDim reports the fitted width."""
    rng = np.random.default_rng(seed=3)
    states = rng.random((20, 7))
    targets = rng.random((20, 2))

    ridge = _rclib.RidgeReadout(alpha=0.5, include_bias=True, tolerance=1e-7)
    assert (ridge.getAlpha(), ridge.getTolerance(), ridge.getInputDim()) == (0.5, 1e-7, 0)
    ridge.fit(states, targets)
    assert ridge.getInputDim() == states.shape[1]

    rls = _rclib.RlsReadout(lambda_=0.95, delta=2.0, include_bias=False)
    assert (rls.getLambda(), rls.getDelta(), rls.getIncludeBias(), rls.getInputDim()) == (0.95, 2.0, False, 0)
    rls.fit(states, targets)
    assert rls.getInputDim() == states.shape[1]

    lms = _rclib.LmsReadout(learning_rate=0.05, include_bias=True)
    assert (lms.getLearningRate(), lms.getIncludeBias(), lms.getInputDim()) == (0.05, True, 0)
    lms.fit(states, targets)
    assert lms.getInputDim() == states.shape[1]
