"""Quick start example."""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, readouts, reservoirs


def main() -> None:
    """Run the quick start example."""
    # 1. Create some dummy data
    x_train = np.linspace(0, 1, 100).reshape(-1, 1)
    y_train = np.sin(x_train * 10)

    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = np.sin(x_test * 10)

    # 2. Configure Reservoir
    res = reservoirs.RandomSparse(
        n_neurons=1000, spectral_radius=0.9, sparsity=0.1, leak_rate=0.3, include_bias=True, input_scaling=0.5
    )

    # 3. Configure Readout
    readout = readouts.Ridge(alpha=1e-8, include_bias=True)

    # 4. Configure Model
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # 5. Fit and Predict
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # 6. Plot the results
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test loss (MSE): {mse:.4e}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(x_test, y_test, label="True")
        plt.plot(x_test, y_pred, label="Predicted")
        plt.text(0.05, 0.95, f"MSE: {mse:.4e}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")
        plt.legend()
        if sys.stdout.isatty():
            plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plot.")


if __name__ == "__main__":
    main()
