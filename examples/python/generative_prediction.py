"""Generative prediction example."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from rclib import readouts, reservoirs
from rclib.model import ESN

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# 1. Generate Mackey-Glass time series data
def mackey_glass(
    n_points: int = 2000,
    tau: int = 17,
    delta_t: float = 1.0,  # noqa: ARG001
    x0: ArrayLike | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate Mackey-Glass time series data."""
    rng = np.random.default_rng(seed=seed)

    x = np.empty(n_points)
    if x0 is None:
        x0_val = rng.random(tau)
    else:
        x0_val = np.array(x0)

    # Initial values
    x[:tau] = x0_val[:tau]

    # Generate the series
    for t in range(tau, n_points):
        x_tau = x[t - tau]
        x[t] = x[t - 1] + (0.2 * x_tau) / (1.0 + x_tau**10) - 0.1 * x[t - 1]

    return x


def main() -> None:
    """Run the generative prediction example."""
    print("Running Generative Prediction Example...")

    # Generate data
    data = mackey_glass(n_points=3000)
    data = data.reshape(-1, 1)

    # 2. Split into training and testing sets
    train_len = 1500
    prime_len = 100  # Length of sequence to prime the reservoir

    train_data = data[:train_len]
    test_data = data[train_len:]

    x_train, y_train = train_data[:-1], train_data[1:]
    x_test, y_test = test_data[:-1], test_data[1:]

    # 3. Configure and build the ESN model
    res = reservoirs.RandomSparse(
        n_neurons=500, spectral_radius=1.2, sparsity=0.1, leak_rate=0.3, input_scaling=1.0, include_bias=True
    )

    readout = readouts.Ridge(alpha=1e-8, include_bias=True)

    model = ESN(connection_type="serial")
    model.add_reservoir(res)
    model.set_readout(readout)

    # 4. Train the model
    print("Training the model...")
    washout = 100
    model.fit(x_train, y_train, washout_len=washout)

    # 5. Prime the model and perform generative prediction
    print("Performing generative prediction...")
    prime_data = x_test[:prime_len]
    n_generate_steps = len(x_test) - prime_len

    # The model's internal state is now primed.
    # We can now call predict_generative.
    # Note: The generative prediction starts *after* the last priming step.
    model.predict(prime_data, reset_state_before_predict=True)  # Prime the state
    generated_output = model.predict_generative(np.array([[]]).reshape(0, 1), n_generate_steps)

    # 6. Visualize the results
    print("Plotting results...")
    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Full test sequence (ground truth)
    full_test_y = y_test.flatten()

    # Generated sequence starts after the priming period
    generated_y = generated_output.flatten()

    time_axis = np.arange(len(full_test_y))

    ax.plot(time_axis, full_test_y, "b", label="Ground Truth")
    # Plot the generated part
    ax.plot(time_axis[prime_len:], generated_y, "r--", label="Generative Prediction")
    # Mark the priming region
    ax.axvspan(0, prime_len, color="gray", alpha=0.2, label="Priming Phase")

    ax.set_title("Mackey-Glass Time Series: Generative Prediction")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(visible=True)

    plt.tight_layout()
    plt.savefig("generative_prediction_result.png")
    print("Plot saved to generative_prediction_result.png")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
