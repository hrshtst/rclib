"""Tests for adaptive Ridge solver selection."""

from __future__ import annotations

import numpy as np
from rclib import ESN, _rclib, readouts, reservoirs


def test_adaptive_solver_small() -> None:
    """Test that a small problem uses the CHOLESKY solver."""
    model = ESN()
    # Total neurons < 8000
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=1000, spectral_radius=0.9))
    model.set_readout(readouts.Ridge(alpha=1e-8, include_bias=True, solver="auto"))

    # Trigger fit to let C++ decide
    rng = np.random.default_rng(seed=42)
    x = rng.random((10, 1))
    y = rng.random((10, 1))
    model.fit(x, y)

    cpp_readout = model._cpp_model.getReadout()  # noqa: SLF001
    assert cpp_readout.getSolver() == _rclib.RidgeReadout.Solver.AUTO
    assert cpp_readout.getEffectiveSolver() == _rclib.RidgeReadout.Solver.CHOLESKY


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
