"""Unit tests for seed consistency in rclib."""

from __future__ import annotations

import numpy as np
from rclib import reservoirs


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
