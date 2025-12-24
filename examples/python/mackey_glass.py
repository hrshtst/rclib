"""Mackey-Glass example."""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, readouts, reservoirs


def mackey_glass(n_samples: int = 1500, tau: int = 17, seed: int = 0) -> np.ndarray:
    """Generate Mackey-Glass time series."""
    # Mackey-Glass time series generation
    rng = np.random.default_rng(seed=seed)
    x = np.zeros(n_samples + tau)
    x[0:tau] = 0.5 + 0.5 * rng.random(tau)
    for t in range(tau, n_samples + tau - 1):
        x[t + 1] = x[t] + (0.2 * x[t - tau]) / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    return x[tau:]


def main() -> None:
    """Run the example."""
    # 1. Generate Mackey-Glass data
    data = mackey_glass()
    x_data = data[:-1].reshape(-1, 1)
    y_data = data[1:].reshape(-1, 1)

    # Split into training and testing sets
    train_len = 1000
    washout_len_val = 100
    x_train, y_train = x_data[:train_len], y_data[:train_len]
    x_test, y_test = x_data[train_len:], y_data[train_len:]

    # 2. Configure Reservoir
    res = reservoirs.RandomSparse(
        n_neurons=2000, spectral_radius=1.1, sparsity=0.05, leak_rate=0.1, include_bias=True, input_scaling=0.5
    )

    # 3. Configure Readout
    readout = readouts.Ridge(alpha=1e-8, include_bias=True)

    # 4. Configure Model
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # 5. Fit and Predict
    model.fit(x_train, y_train, washout_len_val)
    y_pred = model.predict(x_test, reset_state_before_predict=False)

    # 6. Plot the results
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test loss (MSE): {mse:.4e}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 6))
        plt.plot(range(len(y_test)), y_test, label="True")
        plt.plot(range(len(y_pred)), y_pred, label="Predicted")
        plt.text(0.05, 0.95, f"MSE: {mse:.4e}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")
        plt.legend()
        if sys.stdout.isatty():
            plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plot.")


if __name__ == "__main__":
    main()
