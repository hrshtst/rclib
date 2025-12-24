"""Integration tests for online learning in rclib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rclib import ESN, readouts, reservoirs

if TYPE_CHECKING:
    from rclib.readouts import Lms, Rls


def test_online_learning_adaptation() -> None:
    """Verify that ESN can adapt online to a changing signal using LMS and RLS."""
    # 1. Generate data: Sine wave that changes frequency halfway
    n_steps = 600
    change_point = 300
    t = np.linspace(0, 30, n_steps)

    # Frequency 1 then Frequency 2
    signal = np.concatenate(
        [np.sin(t[:change_point] * 2 * np.pi * 0.5), np.sin(t[change_point:] * 2 * np.pi * 1.2)]
    ).reshape(-1, 1)

    x_data = signal[:-1]
    y_data = signal[1:]

    def run_online_experiment(readout_obj: Lms | Rls) -> float:
        # Use a constant seed for reservoir weights initialization
        res = reservoirs.RandomSparse(n_neurons=100, spectral_radius=0.9, include_bias=True, seed=42)
        model = ESN()
        model.add_reservoir(res)
        model.set_readout(readout_obj)

        # Initial fit on first half
        model.fit(x_data[:100], y_data[:100], washout_len=10)

        # Online adaptation on the rest
        errors = []
        for i in range(100, len(x_data)):
            curr_x = x_data[i : i + 1]
            curr_y = y_data[i : i + 1]

            pred = model.predict_online(curr_x)
            errors.append(float(np.mean((pred - curr_y) ** 2)))
            model.partial_fit(curr_x, curr_y)

        return float(np.mean(errors))

    # Thresholds for passing tests
    lms_mse_threshold = 0.1
    rls_mse_threshold = 0.05

    # 2. Test LMS
    lms_mse = run_online_experiment(readouts.Lms(learning_rate=0.01, include_bias=True))
    assert lms_mse < lms_mse_threshold, f"LMS adaptation failed with MSE {lms_mse}"

    # 3. Test RLS
    rls_mse = run_online_experiment(readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True))
    assert rls_mse < rls_mse_threshold, f"RLS adaptation failed with MSE {rls_mse}"
