# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Load the frozen fixtures of model format version 1.

The fixtures in tests/data/serialization/v1 (see generate.py there) check that
files written by earlier builds keep loading with the same configuration and
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rclib import ESN, readouts, reservoirs

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "serialization" / "v1"


def _read_expected(name: str) -> dict[str, np.ndarray]:
    """Parse "<name> <values...>" lines into one-column arrays, skipping comments."""
    arrays = {}
    for line in (FIXTURES / f"{name}.expected.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            key, *values = line.split()
            arrays[key] = np.array([float(value) for value in values]).reshape(-1, 1)
    return arrays


def _configs(esn: ESN) -> list[tuple[type, dict[str, object]]]:
    params = [*esn._reservoirs_params, esn._readout_params]  # noqa: SLF001
    return [(type(config), vars(config)) for config in params]


def _expected_configs(*configs: object) -> list[tuple[type, dict[str, object]]]:
    return [(type(config), vars(config)) for config in configs]


FIXTURE_CONFIGS = {
    "serial_rs_nvar_ridge": (
        "serial",
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.9,
                sparsity=0.5,
                leak_rate=0.5,
                input_scaling=1.0,
                include_bias=True,
                seed=1,
            ),
            reservoirs.Nvar(num_lags=2, polynomial_order=2),
            readouts.Ridge(alpha=1e-3, include_bias=True),
        ),
    ),
    "parallel_rs_nvar_rls": (
        "parallel",
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.8,
                sparsity=0.4,
                leak_rate=0.7,
                input_scaling=0.5,
                include_bias=False,
                seed=2,
            ),
            reservoirs.Nvar(num_lags=2, polynomial_order=1),
            readouts.Rls(lambda_=1.0, delta=0.5, include_bias=True, solver="rank_k_update"),
        ),
    ),
    "serial_rs_lms": (
        "serial",
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.9,
                sparsity=0.5,
                leak_rate=0.5,
                input_scaling=1.0,
                include_bias=True,
                seed=3,
            ),
            readouts.Lms(learning_rate=0.05, include_bias=True),
        ),
    ),
    "unfitted_ridge": (
        "serial",
        _expected_configs(
            reservoirs.RandomSparse(n_neurons=6, spectral_radius=0.9, seed=4),
            readouts.Ridge(alpha=0.5, include_bias=False, solver="cholesky"),
        ),
    ),
}


@pytest.mark.parametrize("name", FIXTURE_CONFIGS)
def test_fixture_configuration(name: str) -> None:
    """The configuration is restored exactly."""
    connection_type, configs = FIXTURE_CONFIGS[name]
    esn = ESN.load(FIXTURES / f"{name}.rclib")
    assert esn.connection_type == connection_type
    assert _configs(esn) == configs


@pytest.mark.parametrize(
    ("name", "online_readout"),
    [("serial_rs_nvar_ridge", False), ("parallel_rs_nvar_rls", True), ("serial_rs_lms", True)],
)
def test_fixture_results(name: str, *, online_readout: bool) -> None:
    """Results follow the sequence documented in generate.py.

    The fixtures are small and numerically stable, so results computed by another
    build agree to a tight tolerance. This is not a general portability bound.
    """
    esn = ESN.load(FIXTURES / f"{name}.rclib")
    expected = _read_expected(name)

    def check(actual: np.ndarray, key: str) -> None:
        np.testing.assert_allclose(actual, expected[key], rtol=1e-10, atol=1e-12)

    check(esn.predict_online(expected["input_online"]), "online")
    check(esn.predict(expected["input_predict"]), "predict")
    if online_readout:
        esn.partial_fit(expected["input_partial_fit"], expected["target_partial_fit"])
        check(esn.predict(expected["input_predict"]), "after_partial_fit")


def test_unfitted_fixture_cannot_predict() -> None:
    """The unfitted fixture loads with an unfitted readout."""
    esn = ESN.load(FIXTURES / "unfitted_ridge.rclib")
    with pytest.raises(RuntimeError, match="must be fit"):
        esn.predict(np.ones((3, 1)))
