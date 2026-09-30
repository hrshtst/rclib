# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for seed consistency in rclib."""

from __future__ import annotations

import numpy as np
from rclib import readouts, reservoirs
from rclib.model import ESN


def test_random_sparse_reservoir_seed_consistency() -> None:
    """Verify that same seed produces identical results and different seeds produce different results."""
    params = {
        "n_neurons": 50,
        "spectral_radius": 0.9,
        "sparsity": 0.1,
        "leak_rate": 0.5,
        "input_scaling": 1.0,
        "include_bias": True,
        "seed": 123,
    }

    # 1. Create two reservoirs with the same seed
    res1 = reservoirs.RandomSparse(**params)
    res2 = reservoirs.RandomSparse(**params)

    # Python-side objects just store parameters, we need to check the C++ core behavior
    # but the current Python API doesn't expose W_res or W_in directly.
    # We can check by comparing the state after one advance() call.

    from rclib import _rclib

    cpp_res1 = _rclib.RandomSparseReservoir(
        params["n_neurons"],
        params["spectral_radius"],
        params["sparsity"],
        params["leak_rate"],
        params["input_scaling"],
        params["include_bias"],
        params["seed"],
    )
    cpp_res2 = _rclib.RandomSparseReservoir(
        params["n_neurons"],
        params["spectral_radius"],
        params["sparsity"],
        params["leak_rate"],
        params["input_scaling"],
        params["include_bias"],
        params["seed"],
    )

    input_data = np.ones((1, 5))

    state1 = cpp_res1.advance(input_data)
    state2 = cpp_res2.advance(input_data)

    # States should be exactly identical
    assert np.array_equal(state1, state2)

    # 2. Create one with a different seed
    cpp_res3 = _rclib.RandomSparseReservoir(
        params["n_neurons"],
        params["spectral_radius"],
        params["sparsity"],
        params["leak_rate"],
        params["input_scaling"],
        params["include_bias"],
        456,
    )
    state3 = cpp_res3.advance(input_data)

    # States should be different
    assert not np.array_equal(state1, state3)


def _fitted_esn(x: np.ndarray, y: np.ndarray) -> ESN:
    """Build an ESN whose reservoir is fully determined by its seed and fit it."""
    model = ESN()
    model.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=300,
            spectral_radius=0.5,
            sparsity=0.1,
            leak_rate=0.3,
            input_scaling=1.0,
            include_bias=True,
            seed=0,
        )
    )
    model.set_readout(readouts.Ridge(alpha=1.0, include_bias=True))
    model.fit(x, y)
    return model


def test_esn_same_configuration_predicts_identically() -> None:
    """Verify that ESNs built one after another in a process from the same configuration predict identically."""
    series = np.sin(0.3 * np.arange(201)).reshape(-1, 1)
    x, y = series[:-1], series[1:]

    model1 = _fitted_esn(x, y)
    model2 = _fitted_esn(x, y)

    np.testing.assert_array_equal(model1.predict(x), model2.predict(x))
    np.testing.assert_array_equal(model1.predict_generative(x[-10:], 50), model2.predict_generative(x[-10:], 50))
