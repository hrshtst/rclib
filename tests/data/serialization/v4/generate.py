# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Generate the frozen fixtures of rclib model format version 4.

Do not regenerate these files. They pin format version 4: the C++ and Python
test suites load them to check that files written by earlier builds keep
loading with the same configuration and results. A format change must add a new
version directory with its own generator and fixtures, and keep these fixtures
and their tests.

Version 4 added the reward-modulated readout and reservoir and models without a
readout. The first six fixtures repeat the configurations of version 3. The four
``reward_*`` fixtures hold reward-modulated components saved in the middle of
learning, with pending eligibility, post-synaptic averages, baselines of several
keys and advanced noise generators.

The files were generated once with::

    uv run python tests/data/serialization/v4/generate.py

Each ``<name>.expected.txt`` holds one array per line, a name followed by its
values written with ``%.17g`` so they round-trip exactly. Inputs are stored too,
so the tests do not depend on how a math library evaluates them. The expected
results come from this sequence on the loaded model:

1. ``online = predict_online(input_online)``, continuing from the saved states;
2. ``predict = predict(input_predict)``, which resets the states first;
3. online readouts only: ``partial_fit(input_partial_fit, target_partial_fit)``,
   then ``after_partial_fit = predict(input_predict)``.

The reward fixtures use this sequence instead, which pins the continuation of the
noise and the learning state:

1. ``explore = act(input_act)``, exploring from the saved state;
2. ``reinforce(reward[0], key=int(key[0]))``;
3. ``noise_free = act(input_noise_free, explore=False)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rclib import ESN, learning, readouts, reservoirs

HERE = Path(__file__).resolve().parent


def _signal(n_samples: int, phase: float) -> np.ndarray:
    return np.sin(0.3 * np.arange(n_samples) + phase).reshape(-1, 1)


def _write_expected(name: str, arrays: dict[str, np.ndarray]) -> None:
    lines = [
        f"# Expected results for {name}.rclib (rclib model format v4). Do not regenerate.",
        "# One array per line: a name, then its values written with %.17g.",
    ]
    lines += [" ".join([key, *(f"{value:.17g}" for value in np.ravel(values))]) for key, values in arrays.items()]
    (HERE / f"{name}.expected.txt").write_text("\n".join(lines) + "\n")


def _save_with_expectations(name: str, esn: ESN, *, online_readout: bool) -> None:
    path = HERE / f"{name}.rclib"
    esn.save(path)

    loaded = ESN.load(path)
    input_online = _signal(5, 2.0)
    input_predict = _signal(15, 1.0)
    arrays = {
        "input_online": input_online,
        "online": loaded.predict_online(input_online),
        "input_predict": input_predict,
        "predict": loaded.predict(input_predict),
    }
    if online_readout:
        series = _signal(3, 3.0)
        arrays["input_partial_fit"] = series[:2]
        arrays["target_partial_fit"] = series[1:]
        loaded.partial_fit(series[:2], series[1:])
        arrays["after_partial_fit"] = loaded.predict(input_predict)
    _write_expected(name, arrays)


def _save_reward_fixture(name: str, esn: ESN, *, input_width: int, key: int) -> None:
    path = HERE / f"{name}.rclib"
    esn.save(path)

    loaded = ESN.load(path)
    input_act = np.hstack([_signal(4, 0.5 + k) for k in range(input_width)])
    input_noise_free = np.hstack([_signal(5, 1.5 + k) for k in range(input_width)])
    explore = loaded.act(input_act)
    reward = np.array([-0.35])
    loaded.reinforce(float(reward[0]), key=key)
    arrays = {
        "input_act": input_act,
        "explore": explore,
        "reward": reward,
        "key": np.array([float(key)]),
        "input_noise_free": input_noise_free,
        "noise_free": loaded.act(input_noise_free, explore=False),
    }
    _write_expected(name, arrays)


def _learn(esn: ESN, *, input_width: int, trials: int, keys: int, trial_length: int = 3) -> None:
    """Explore and reward a few trials with fixed rewards, then leave a trial pending."""
    for trial in range(trials):
        inputs = np.hstack([_signal(trial_length, 0.2 * trial + k) for k in range(input_width)])
        esn.act(inputs)
        esn.reinforce(0.1 * trial - 0.2, key=trial % keys)
    esn.act(np.hstack([_signal(2, 3.0 + k) for k in range(input_width)]))


def _write_reward_fixtures() -> None:
    """Write the reward-modulated fixtures of version 4."""
    # Serial RandomSparse with an online reward-modulated readout: Heaviside modulator,
    # first-difference post signal and a decaying learning rate.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8, spectral_radius=0.9, sparsity=0.5, leak_rate=0.5, input_scaling=1.0, include_bias=True, seed=7
        )
    )
    esn.set_readout(
        readouts.RewardModulated(
            2,
            learning.hoerzer2014(0.05, learning_rate_decay_steps=20.0, exploration=learning.Exploration(seed=11)),
            include_bias=True,
        )
    )
    _learn(esn, input_width=1, trials=4, keys=2)
    _save_reward_fixture("reward_serial_rs_readout", esn, input_width=1, key=1)

    # A reward-modulated reservoir learning by trials with a supralinear eligibility, masks
    # and windowed baselines of three keys, and a Ridge readout fitted before learning.
    esn = ESN("serial")
    esn.add_reservoir(
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
        )
    )
    esn.set_readout(readouts.Ridge(alpha=1e-3, include_bias=True))
    series = _signal(31, 0.0)
    esn.fit(series[:-1], series[1:], washout_len=5)
    _learn(esn, input_width=1, trials=7, keys=3)
    _save_reward_fixture("reward_reservoir_ridge", esn, input_width=1, key=2)

    # An identity NVAR with a decorrelated trial readout and Gaussian noise.
    esn = ESN("serial")
    esn.add_reservoir(reservoirs.Nvar(num_lags=1, polynomial_order=1))
    esn.set_readout(
        readouts.RewardModulated(
            1,
            learning.burms2015(0.2, exploration=learning.Exploration(distribution="gaussian", amplitude=0.1, seed=13)),
            include_bias=False,
        )
    )
    _learn(esn, input_width=3, trials=4, keys=1)
    _save_reward_fixture("reward_nvar_decorrelated", esn, input_width=3, key=0)

    # A reward-modulated reservoir with no readout, learning online with the EH rule.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RewardModulated(
            6,
            1.1,
            0.6,
            0.5,
            1.0,
            rule=learning.legenstein2010(0.05, exploration=learning.Exploration(seed=14)),
            include_bias=True,
            seed=9,
        )
    )
    _learn(esn, input_width=2, trials=3, keys=2)
    _save_reward_fixture("reward_reservoir_no_readout", esn, input_width=2, key=0)


def main() -> None:
    """Write the v4 fixtures next to this script."""
    series = _signal(61, 0.0)

    # Serial RandomSparse -> NVAR with Ridge; AUTO resolves to DUAL_CHOLESKY
    # because the 152 NVAR features outnumber the 55 training samples.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8, spectral_radius=0.9, sparsity=0.5, leak_rate=0.5, input_scaling=1.0, include_bias=True, seed=1
        )
    )
    esn.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=2))
    esn.set_readout(readouts.Ridge(alpha=1e-3, include_bias=True))
    esn.fit(series[:-1], series[1:], washout_len=5)
    _save_with_expectations("serial_rs_nvar_ridge", esn, online_readout=False)

    # Parallel RandomSparse + NVAR with rank-k RLS (lambda = 1 enables the Woodbury update).
    esn = ESN("parallel")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8, spectral_radius=0.8, sparsity=0.4, leak_rate=0.7, input_scaling=0.5, include_bias=False, seed=2
        )
    )
    esn.add_reservoir(reservoirs.Nvar(num_lags=2, polynomial_order=1))
    esn.set_readout(readouts.Rls(lambda_=1.0, delta=0.5, include_bias=True, solver="rank_k_update"))
    esn.fit(series[:40], series[1:41])
    _save_with_expectations("parallel_rs_nvar_rls", esn, online_readout=True)

    # Serial RandomSparse with LMS.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8, spectral_radius=0.9, sparsity=0.5, leak_rate=0.5, input_scaling=1.0, include_bias=True, seed=3
        )
    )
    esn.set_readout(readouts.Lms(learning_rate=0.05, include_bias=True))
    esn.fit(series[:40], series[1:41])
    _save_with_expectations("serial_rs_lms", esn, online_readout=True)

    # Never used: W_in is not generated and the readout is unfitted, so this pins
    # the encoding of uninitialized components. It has no expected results.
    esn = ESN("serial")
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=6, spectral_radius=0.9, seed=4))
    esn.set_readout(readouts.Ridge(alpha=0.5, include_bias=False, solver="cholesky"))
    esn.save(HERE / "unfitted_ridge.rclib")

    # Serial RandomSparse scaled by the dense method, with Ridge.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8,
            spectral_radius=0.9,
            sparsity=0.5,
            leak_rate=0.5,
            input_scaling=1.0,
            include_bias=True,
            seed=5,
            spectral_radius_method="dense",
        )
    )
    esn.set_readout(readouts.Ridge(alpha=1e-3, include_bias=True))
    esn.fit(series[:-1], series[1:], washout_len=5)
    _save_with_expectations("serial_rs_dense_ridge", esn, online_readout=False)

    # Serial RandomSparse with its bias scaled by 0.25, with Ridge.
    esn = ESN("serial")
    esn.add_reservoir(
        reservoirs.RandomSparse(
            n_neurons=8,
            spectral_radius=0.9,
            sparsity=0.5,
            leak_rate=0.5,
            input_scaling=1.0,
            include_bias=True,
            seed=6,
            bias_scaling=0.25,
        )
    )
    esn.set_readout(readouts.Ridge(alpha=1e-3, include_bias=True))
    esn.fit(series[:-1], series[1:], washout_len=5)
    _save_with_expectations("serial_rs_bias_ridge", esn, online_readout=False)

    _write_reward_fixtures()


if __name__ == "__main__":
    main()
