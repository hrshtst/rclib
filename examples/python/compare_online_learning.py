"""Example script comparing online learning methods."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from rclib import ESN, readouts, reservoirs


def main() -> None:
    """
    An example script demonstrating online learning with LMS and RLS.

    The task is to adapt to a changing sine wave frequency using the rclib library.
    """
    # --- 1. Configuration (Simplified) ---
    print("--- Configuration ---")
    n_neurons = 2000
    spectral_radius = 0.99
    sparsity = 0.02
    leak_rate = 0.2
    include_bias = True

    ridge_alpha = 1e-4
    lms_learning_rate = 0.001
    rls_lambda = 0.999
    rls_delta = 0.1

    plot_output_file = "online_learning_adaptation_rclib.png"

    # --- 2. Generate Data with Changing Dynamics ---
    print("--- Generating Data ---")
    n_total = 4000
    n_train = 2000
    change_point = 3000

    time_vec = np.linspace(0, 80, n_total)
    freq1, freq2 = 1.0, 1.5

    signal = np.zeros(n_total)
    signal[:change_point] = np.sin(freq1 * time_vec[:change_point])
    signal[change_point:] = np.sin(freq2 * time_vec[change_point:])

    data = signal.reshape(-1, 1).astype(np.float64)  # Use float64 for Eigen compatibility

    train_input = data[: n_train - 1]
    train_target = data[1:n_train]
    test_input = data[n_train - 1 : -1]
    test_target = data[n_train:]

    # --- 3. Offline Training (Ridge) ---
    print("\n--- 1. Offline Training with Ridge ---")
    res_ridge = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=1.0,
        include_bias=include_bias,
    )
    readout_ridge = readouts.Ridge(alpha=ridge_alpha, include_bias=include_bias)
    esn_ridge = ESN()
    esn_ridge.add_reservoir(res_ridge)
    esn_ridge.set_readout(readout_ridge)
    esn_ridge.fit(train_input, train_target)

    print("\n--- 2. Predicting with offline-trained model (no adaptation) ---")
    preds_ridge = esn_ridge.predict(test_input)

    # --- 4. Online Adaptation (LMS) ---
    print("\n--- 3. Adapting online with LMS ---")
    res_lms = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=1.0,
        include_bias=include_bias,
    )
    readout_lms = readouts.Lms(learning_rate=lms_learning_rate, include_bias=include_bias)
    esn_lms = ESN()
    esn_lms.add_reservoir(res_lms)
    esn_lms.set_readout(readout_lms)

    # Online adaptation loop for LMS
    preds_lms = np.zeros_like(test_target)
    # Initialize W_out by performing one partial_fit before the main loop
    esn_lms.partial_fit(test_input[0:1, :], test_target[0:1, :])  # Call partial_fit once
    for i in range(test_input.shape[0]):
        current_input = test_input[i : i + 1, :]
        current_target = test_target[i : i + 1, :]

        # Predict current step
        pred_lms_step = esn_lms.predict_online(current_input)
        preds_lms[i] = pred_lms_step

        # Adapt model using partial_fit
        esn_lms.partial_fit(current_input, current_target)

    # --- 5. Online Adaptation (RLS) ---
    print("\n--- 4. Adapting online with RLS ---")
    res_rls = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=1.0,
        include_bias=include_bias,
    )
    readout_rls = readouts.Rls(lambda_=rls_lambda, delta=rls_delta, include_bias=include_bias)
    esn_rls = ESN()
    esn_rls.add_reservoir(res_rls)
    esn_rls.set_readout(readout_rls)

    # Online adaptation loop for RLS
    preds_rls = np.zeros_like(test_target)
    # Initialize W_out by performing one partial_fit before the main loop
    esn_rls.partial_fit(test_input[0:1, :], test_target[0:1, :])  # Call partial_fit once
    for i in range(test_input.shape[0]):
        current_input = test_input[i : i + 1, :]
        current_target = test_target[i : i + 1, :]

        # Predict current step
        pred_rls_step = esn_rls.predict_online(current_input)
        preds_rls[i] = pred_rls_step

        # Adapt model using partial_fit
        esn_rls.partial_fit(current_input, current_target)

    # --- 6. Plot Results ---
    print("\n--- Plotting results ---")
    plt.figure(figsize=(15, 8))
    plt.plot(test_target, "k", label="True Signal", linewidth=2)
    plt.plot(preds_ridge, "b--", label="Ridge (No Adaptation)", alpha=0.8)
    plt.plot(preds_lms, "g-.", label="LMS (Online Adaptation)", alpha=0.8)
    plt.plot(preds_rls, "r:", label="RLS (Online Adaptation)", alpha=0.8)

    change_idx = change_point - n_train
    plt.axvline(x=change_idx, color="gray", linestyle="--", label="Frequency Change")

    plt.title("ESN Online Learning: Adapting to Changing Sine Wave Frequency (rclib)", fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(visible=True)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.savefig(plot_output_file)
    print(f"Plot saved to {plot_output_file}")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
