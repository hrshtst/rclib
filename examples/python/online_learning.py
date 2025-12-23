"""Online learning example."""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, readouts, reservoirs


def main() -> None:
    """Run the online learning example."""
    # 1. Create some dummy data for online training
    num_steps = 1000
    input_dim = 1
    output_dim = 1

    x_data = np.array([np.sin(i / 100.0) for i in range(num_steps)]).reshape(-1, input_dim)
    y_data = np.array([np.cos(i / 100.0) for i in range(num_steps)]).reshape(-1, output_dim)

    # 2. Configure Reservoir
    res = reservoirs.RandomSparse(
        n_neurons=100,  # Smaller reservoir for online example
        spectral_radius=0.9,
        sparsity=0.1,
        leak_rate=0.3,
        include_bias=True,
    )

    # 3. Configure Readout for online learning (RLS)
    readout = readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True)

    # 4. Configure Model
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # 5. Online Training Loop
    print("Starting online training...")
    for i in range(num_steps):
        current_input = x_data[i : i + 1, :]
        current_target = y_data[i : i + 1, :]

        model.partial_fit(current_input, current_target)

        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{num_steps}")
    print("Online training finished.")

    # 6. Predict on some new data (e.g., the same data for evaluation)
    y_pred = model.predict(x_data)

    # 7. Print the results
    mse = np.mean((y_pred - y_data) ** 2)
    print(f"Test loss (MSE) after online training: {mse:.4e}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(num_steps), y_data, label="True")
        plt.plot(np.arange(num_steps), y_pred, label="Predicted")
        plt.title("Online Learning Example")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        if sys.stdout.isatty():
            plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plot.")


if __name__ == "__main__":
    main()
