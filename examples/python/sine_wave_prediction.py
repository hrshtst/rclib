import matplotlib.pyplot as plt
import numpy as np
from rcl.model import ESN
from rcl.readouts import Ridge
from rcl.reservoirs import RandomSparse


def run_experiment(include_bias: bool, n_total_samples: int, n_train_samples: int, noise_amplitude: float,
                   n_neurons: int, spectral_radius: float, sparsity: float, leak_rate: float,
                   ridge_alpha: float, plot_filename: str):
    print(f"--- Running experiment with include_bias={include_bias} ---")

    # --- 1. Data Generation ---
    print("--- Generating Data ---")
    time_np = np.linspace(0, 80, n_total_samples)
    clean_data = np.sin(time_np)
    noise = noise_amplitude * np.random.randn(n_total_samples)
    data = (clean_data + noise).reshape(-1, 1).astype(np.float64)  # Use float64 for rcl

    input_data, target_data = data[:-1], data[1:]
    train_input = input_data[:n_train_samples]
    train_target = target_data[:n_train_samples]
    test_input = data[n_train_samples:-1]
    test_target = data[n_train_samples + 1:]

    # --- 2. Instantiate, Train, and Predict ---
    print("--- Initializing ESN ---")
    reservoir = RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        include_bias=include_bias,
    )

    readout = Ridge(alpha=ridge_alpha, include_bias=include_bias)

    model = ESN(connection_type="serial")
    model.add_reservoir(reservoir)
    model.set_readout(readout)

    print("--- Fitting ESN ---")
    model.fit(train_input, train_target)

    print("--- Predicting with ESN ---")
    predictions = model.predict(test_input)

    mse = np.mean((predictions[:len(test_target)] - test_target) ** 2)
    print(f"Mean Squared Error (include_bias={include_bias}): {mse:.6f}")

    # --- 3. Plot Results ---
    print(f"\nPlotting results for include_bias={include_bias}...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_range = range(min(200, len(test_target)))
    ax.plot(test_target[plot_range], "b", label="True Target (with noise)", linewidth=2, alpha=0.7)
    ax.plot(predictions[plot_range], "r--", label="ESN Prediction", linewidth=2)
    ax.set_title(f"Echo State Network: Noisy Sine Wave Prediction (Test Set, Bias={include_bias})", fontsize=16)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.text(
        0.02,
        0.1,
        f"MSE: {mse:.6f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )
    plt.tight_layout()

    plt.savefig(plot_filename)
    print(f"Plot saved to '{plot_filename}'")
    plt.close(fig) # Close the figure to free memory

    return mse


def main():
    # --- Configuration Parameters ---
    n_total_samples = 1000
    n_train_samples = 800
    noise_amplitude = 0.05

    n_neurons = 2000
    spectral_radius = 0.99
    sparsity = 0.02
    leak_rate = 0.2
    ridge_alpha = 1e-4

    # Run with bias
    mse_with_bias = run_experiment(
        include_bias=True,
        n_total_samples=n_total_samples,
        n_train_samples=n_train_samples,
        noise_amplitude=noise_amplitude,
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        ridge_alpha=ridge_alpha,
        plot_filename="sine_wave_prediction_with_bias.png"
    )

    # Run without bias
    mse_without_bias = run_experiment(
        include_bias=False,
        n_total_samples=n_total_samples,
        n_train_samples=n_train_samples,
        noise_amplitude=noise_amplitude,
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        ridge_alpha=ridge_alpha,
        plot_filename="sine_wave_prediction_without_bias.png"
    )

    print("\n--- Comparison of Results ---")
    print(f"MSE with bias: {mse_with_bias:.6f}")
    print(f"MSE without bias: {mse_without_bias:.6f}")

    if mse_with_bias < mse_without_bias:
        print("Conclusion: Model performed better with bias.")
    elif mse_without_bias < mse_with_bias:
        print("Conclusion: Model performed better without bias.")
    else:
        print("Conclusion: Model performance was similar with and without bias.")


if __name__ == "__main__":
    main()
