# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Load the frozen fixtures of model format versions 1 and 2.

The fixtures in tests/data/serialization/v<N> (see generate.py there) check that
files written by earlier builds keep loading with the same configuration and
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rclib import ESN, readouts, reservoirs

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "serialization"


def _fixture(version: int, name: str) -> Path:
    return FIXTURES / f"v{version}" / f"{name}.rclib"


def _read_expected(version: int, name: str) -> dict[str, np.ndarray]:
    """Parse "<name> <values...>" lines into one-column arrays, skipping comments."""
    arrays = {}
    for line in (FIXTURES / f"v{version}" / f"{name}.expected.txt").read_text().splitlines():
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
# Fixtures that exist from format version 2 on.
V2_FIXTURE_CONFIGS = {
    "serial_rs_dense_ridge": (
        "serial",
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.9,
                sparsity=0.5,
                leak_rate=0.5,
                input_scaling=1.0,
                include_bias=True,
                seed=5,
                spectral_radius_method="dense",
            ),
            readouts.Ridge(alpha=1e-3, include_bias=True),
        ),
    ),
}
FIXTURES_BY_VERSION = [(1, name) for name in FIXTURE_CONFIGS] + [
    (2, name) for name in [*FIXTURE_CONFIGS, *V2_FIXTURE_CONFIGS]
]
# Fixtures with expected results, and whether their readout learns online.
RESULTS = {
    "serial_rs_nvar_ridge": False,
    "parallel_rs_nvar_rls": True,
    "serial_rs_lms": True,
    "serial_rs_dense_ridge": False,
}


@pytest.mark.parametrize(("version", "name"), FIXTURES_BY_VERSION)
def test_fixture_configuration(version: int, name: str) -> None:
    """The configuration is restored exactly; version 1 files load as power iteration."""
    connection_type, configs = (FIXTURE_CONFIGS | V2_FIXTURE_CONFIGS)[name]
    esn = ESN.load(_fixture(version, name))
    assert esn.connection_type == connection_type
    assert _configs(esn) == configs


@pytest.mark.parametrize(("version", "name"), [fixture for fixture in FIXTURES_BY_VERSION if fixture[1] in RESULTS])
def test_fixture_results(version: int, name: str) -> None:
    """Results follow the sequence documented in generate.py.

    The fixtures are small and numerically stable, so results computed by another
    build agree to a tight tolerance. This is not a general portability bound.
    """
    esn = ESN.load(_fixture(version, name))
    expected = _read_expected(version, name)

    def check(actual: np.ndarray, key: str) -> None:
        np.testing.assert_allclose(actual, expected[key], rtol=1e-10, atol=1e-12)

    check(esn.predict_online(expected["input_online"]), "online")
    check(esn.predict(expected["input_predict"]), "predict")
    if RESULTS[name]:
        esn.partial_fit(expected["input_partial_fit"], expected["target_partial_fit"])
        check(esn.predict(expected["input_predict"]), "after_partial_fit")


@pytest.mark.parametrize("version", [1, 2])
def test_unfitted_fixture_cannot_predict(version: int) -> None:
    """The unfitted fixture loads with an unfitted readout."""
    esn = ESN.load(_fixture(version, "unfitted_ridge"))
    with pytest.raises(RuntimeError, match="must be fit"):
        esn.predict(np.ones((3, 1)))
