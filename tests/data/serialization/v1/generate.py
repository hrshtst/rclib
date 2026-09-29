# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Generate the frozen fixtures of rclib model format version 1.

Do not regenerate these files. They pin format version 1: the C++ and Python
test suites load them to check that files written by earlier builds keep
loading with the same configuration and results. A format change must add a new
version directory with its own generator and fixtures, and keep these fixtures
and their tests.

The files were generated once with::

    uv run python tests/data/serialization/v1/generate.py

Each ``<name>.expected.txt`` holds one array per line, a name followed by its
values written with ``%.17g`` so they round-trip exactly. Inputs are stored too,
so the tests do not depend on how a math library evaluates them. The expected
results come from this sequence on the loaded model:

1. ``online = predict_online(input_online)``, continuing from the saved states;
2. ``predict = predict(input_predict)``, which resets the states first;
3. online readouts only: ``partial_fit(input_partial_fit, target_partial_fit)``,
   then ``after_partial_fit = predict(input_predict)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rclib import ESN, readouts, reservoirs

HERE = Path(__file__).resolve().parent


def _signal(n_samples: int, phase: float) -> np.ndarray:
    return np.sin(0.3 * np.arange(n_samples) + phase).reshape(-1, 1)


def _write_expected(name: str, arrays: dict[str, np.ndarray]) -> None:
    lines = [
        f"# Expected results for {name}.rclib (rclib model format v1). Do not regenerate.",
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


def main() -> None:
    """Write the v1 fixtures next to this script."""
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


if __name__ == "__main__":
    main()
