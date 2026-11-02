# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Load the frozen fixtures of model format versions 1 to 4.

The fixtures in tests/data/serialization/v<N> (see generate.py there) check that
files written by earlier builds keep loading with the same configuration and
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rclib import ESN, learning, readouts, reservoirs

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


def _configs(esn: ESN) -> list[tuple[type, dict[str, object]] | None]:
    params = [*esn._reservoirs_params, esn._readout_params]  # noqa: SLF001
    return [None if config is None else (type(config), vars(config)) for config in params]


def _expected_configs(*configs: object) -> list[tuple[type, dict[str, object]] | None]:
    return [None if config is None else (type(config), vars(config)) for config in configs]


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
# Fixtures that exist from format version 3 on.
V3_FIXTURE_CONFIGS = {
    "serial_rs_bias_ridge": (
        "serial",
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.9,
                sparsity=0.5,
                leak_rate=0.5,
                input_scaling=1.0,
                include_bias=True,
                seed=6,
                bias_scaling=0.25,
            ),
            readouts.Ridge(alpha=1e-3, include_bias=True),
        ),
    ),
}
# Fixtures that exist from format version 4 on: reward-modulated components saved mid-learning,
# with the input width of their expected results.
V4_REWARD_FIXTURES = {
    "reward_serial_rs_readout": (
        "serial",
        1,
        _expected_configs(
            reservoirs.RandomSparse(
                n_neurons=8,
                spectral_radius=0.9,
                sparsity=0.5,
                leak_rate=0.5,
                input_scaling=1.0,
                include_bias=True,
                seed=7,
            ),
            readouts.RewardModulated(
                2,
                learning.hoerzer2014(0.05, learning_rate_decay_steps=20.0, exploration=learning.Exploration(seed=11)),
                include_bias=True,
            ),
        ),
    ),
    "reward_reservoir_ridge": (
        "serial",
        1,
        _expected_configs(
            reservoirs.RewardModulated(
                8,
                0.9,
                0.5,
                0.4,
                1.0,
                rule=learning.miconi2017(
                    0.5,
                    exploration=learning.Exploration(amplitude=0.5, probability=0.2, seed=12),
                    max_update=0.01,
                    baseline=learning.Baseline(kind="window", window=2),
                ),
                include_bias=True,
                seed=8,
                plastic_presynaptic=[0, 1, 2, 3, 4, 5],
                plastic_postsynaptic=[1, 2, 3, 5, 6, 7],
            ),
            readouts.Ridge(alpha=1e-3, include_bias=True),
        ),
    ),
    "reward_nvar_decorrelated": (
        "serial",
        3,
        _expected_configs(
            reservoirs.Nvar(num_lags=1, polynomial_order=1),
            readouts.RewardModulated(
                1,
                learning.burms2015(
                    0.2, exploration=learning.Exploration(distribution="gaussian", amplitude=0.1, seed=13)
                ),
                include_bias=False,
            ),
        ),
    ),
    "reward_reservoir_no_readout": (
        "serial",
        2,
        _expected_configs(
            reservoirs.RewardModulated(
                6,
                1.1,
                0.6,
                0.5,
                1.0,
                rule=learning.legenstein2010(0.05, exploration=learning.Exploration(seed=14)),
                include_bias=True,
                seed=9,
            ),
            None,
        ),
    ),
}
FIXTURES_BY_VERSION = (
    [(1, name) for name in FIXTURE_CONFIGS]
    + [(2, name) for name in [*FIXTURE_CONFIGS, *V2_FIXTURE_CONFIGS]]
    + [(version, name) for version in (3, 4) for name in [*FIXTURE_CONFIGS, *V2_FIXTURE_CONFIGS, *V3_FIXTURE_CONFIGS]]
)
# Fixtures with expected results, and whether their readout learns online.
RESULTS = {
    "serial_rs_nvar_ridge": False,
    "parallel_rs_nvar_rls": True,
    "serial_rs_lms": True,
    "serial_rs_dense_ridge": False,
    "serial_rs_bias_ridge": False,
}


@pytest.mark.parametrize(("version", "name"), FIXTURES_BY_VERSION)
def test_fixture_configuration(version: int, name: str) -> None:
    """The configuration is restored exactly.

    Version 1 files load as power iteration, and version 1 and 2 files with a bias_scaling of 1.
    """
    connection_type, configs = (FIXTURE_CONFIGS | V2_FIXTURE_CONFIGS | V3_FIXTURE_CONFIGS)[name]
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


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_unfitted_fixture_cannot_predict(version: int) -> None:
    """The unfitted fixture loads with an unfitted readout."""
    esn = ESN.load(_fixture(version, "unfitted_ridge"))
    with pytest.raises(RuntimeError, match="must be fit"):
        esn.predict(np.ones((3, 1)))


@pytest.mark.parametrize("name", V4_REWARD_FIXTURES)
def test_reward_fixture_configuration(name: str) -> None:
    """A reward-modulated model saved mid-learning restores its configuration exactly."""
    connection_type, _, configs = V4_REWARD_FIXTURES[name]
    esn = ESN.load(_fixture(4, name))
    assert esn.connection_type == connection_type
    assert _configs(esn) == configs


@pytest.mark.parametrize("name", V4_REWARD_FIXTURES)
def test_reward_fixture_results(name: str) -> None:
    """Exploring, reinforcing and noise-free steps follow the sequence documented in v4/generate.py."""
    _, input_width, _ = V4_REWARD_FIXTURES[name]
    esn = ESN.load(_fixture(4, name))
    expected = _read_expected(4, name)

    def check(actual: np.ndarray, key: str) -> None:
        np.testing.assert_allclose(actual.reshape(-1, 1), expected[key], rtol=1e-10, atol=1e-12)

    check(esn.act(expected["input_act"].reshape(-1, input_width)), "explore")
    esn.reinforce(float(expected["reward"][0, 0]), key=int(expected["key"][0, 0]))
    check(esn.act(expected["input_noise_free"].reshape(-1, input_width), explore=False), "noise_free")
