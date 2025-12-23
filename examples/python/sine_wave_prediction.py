"""Sine wave prediction example."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from rclib.model import ESN
from rclib.readouts import Ridge
from rclib.reservoirs import RandomSparse


def main() -> None:
    """Run the sine wave prediction example."""
    # --- Configuration Parameters ---
    n_total_samples = 1000
    n_train_samples = 800
    noise_amplitude = 0.05

    n_neurons = 2000
    spectral_radius = 0.99
    sparsity = 0.02
    leak_rate = 0.2
    ridge_alpha = 1e-4
    input_scaling = 1.0
    washout_len = 100
    include_bias = True
    reset_state_before_predict = False

    plot_output_file = "sine_wave_prediction.png"

    # --- 1. Data Generation ---
    print("--- Generating Data ---")
    rng = np.random.default_rng(seed=0)
    time_np = np.linspace(0, 80, n_total_samples)
    clean_data = np.sin(time_np)
    noise = noise_amplitude * rng.standard_normal(n_total_samples)
    data = (clean_data + noise).reshape(-1, 1).astype(np.float64)

    input_data, target_data = data[:-1], data[1:]
    train_input = input_data[:n_train_samples]
    train_target = target_data[:n_train_samples]
    test_input = data[n_train_samples:-1]
    test_target = data[n_train_samples + 1 :]

    # --- 2. Instantiate, Train, and Predict ---
    print("--- Initializing ESN ---")
    reservoir = RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=include_bias,
    )

    readout = Ridge(alpha=ridge_alpha, include_bias=include_bias)

    model = ESN(connection_type="serial")
    model.add_reservoir(reservoir)
    model.set_readout(readout)

    print("--- Fitting ESN ---")
    model.fit(train_input, train_target, washout_len=washout_len)

    print("--- Predicting with ESN ---")
    predictions = model.predict(test_input, reset_state_before_predict=reset_state_before_predict)

    mse = np.mean((predictions[: len(test_target)] - test_target) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

    # --- 3. Plot Results ---
    print("\nPlotting results...")
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(15, 6))
    plot_range = range(min(200, len(test_target)))
    ax.plot(test_target[plot_range], "b", label="True Target (with noise)", linewidth=2, alpha=0.7)
    ax.plot(predictions[plot_range], "r--", label="ESN Prediction", linewidth=2)
    ax.set_title("ESN: Sine Wave Prediction", fontsize=16)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.text(
        0.02,
        0.1,
        f"MSE: {mse:.6f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.7},
    )
    plt.tight_layout()
    plt.savefig(plot_output_file)
    print(f"Plot saved to {plot_output_file}")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
