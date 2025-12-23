"""Multi-dimensional prediction example (Lorenz Attractor)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from rclib.model import ESN
from rclib.readouts import Ridge
from rclib.reservoirs import RandomSparse

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def lorenz(xyz: ArrayLike, *, s: float = 10, r: float = 28, b: float = 2.667) -> np.ndarray:
    """Calculate the derivatives of the Lorenz attractor.

    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at `xyz`.
    """
    xyz_arr = np.array(xyz)
    x, y, z = xyz_arr
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def main() -> None:
    """Run the multi-dimensional prediction example."""
    # --- Configuration Parameters ---
    n_total_samples = 2000
    n_train_samples = 1500
    dt = 0.01

    n_neurons = 3000
    spectral_radius = 1.25
    sparsity = 0.05
    leak_rate = 0.3
    ridge_alpha = 1e-5
    input_scaling = 1.0
    washout_len = 200
    include_bias = True
    reset_state_before_predict = False

    plot_output_file = "multi_dimensional_prediction.png"

    # --- 1. Data Generation (Lorenz Attractor) ---
    print("--- Generating Lorenz Attractor Data ---")
    xyzs = np.empty((n_total_samples + 1, 3))
    xyzs[0] = (0.0, 1.0, 1.05)  # Initial condition
    for i in range(n_total_samples):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

    data = xyzs.astype(np.float64)

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

    plot_output_file_3d = "multi_dimensional_prediction_3d.png"

    # --- 3. Plot Results ---
    print("\nPlotting results...")
    plt.style.use("seaborn-v0_8-whitegrid")
    _, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    plot_range = range(min(300, len(test_target)))

    for i, (ax, dim_name) in enumerate(zip(axes, ["X", "Y", "Z"], strict=False)):
        ax.plot(
            test_target[plot_range, i],
            "b",
            label=f"True {dim_name}",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            predictions[plot_range, i],
            "r--",
            label=f"ESN Prediction ({dim_name})",
            linewidth=2,
        )
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    axes[0].set_title(f"ESN: Lorenz Attractor Prediction (MSE: {mse:.6f})", fontsize=16)
    axes[-1].set_xlabel("Time Step")

    plt.tight_layout()
    plt.savefig(plot_output_file)
    print(f"Plot saved to {plot_output_file}")

    # --- 4. 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        test_target[:, 0],
        test_target[:, 1],
        test_target[:, 2],
        "b-",
        label="True Trajectory",
        linewidth=1,
        alpha=0.7,
    )
    ax.plot(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        "r--",
        label="ESN Prediction",
        linewidth=1,
    )
    ax.set_title("Lorenz Attractor: True vs. Predicted Trajectory", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.savefig(plot_output_file_3d)
    print(f"3D plot saved to {plot_output_file_3d}")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
