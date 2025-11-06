import matplotlib.pyplot as plt
import numpy as np
from rcl.model import ESN
from rcl.readouts import Ridge
from rcl.reservoirs import RandomSparse


def run_experiment(include_bias: bool, n_total_samples: int, n_train_samples: int, noise_amplitude: float,
                   n_neurons: int, spectral_radius: float, sparsity: float, leak_rate: float,
                   input_scaling: float, ridge_alpha: float, washout_len: int, reset_state_before_predict: bool, plot_filename: str):
    print(f"--- Running experiment with include_bias={include_bias}, washout_len={washout_len}, input_scaling={input_scaling}, reset_state_before_predict={reset_state_before_predict} ---")

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

    mse = np.mean((predictions[:len(test_target)] - test_target) ** 2)
    print(f"Mean Squared Error (include_bias={include_bias}, washout_len={washout_len}, input_scaling={input_scaling}, reset_state_before_predict={reset_state_before_predict}): {mse:.6f}")

    # --- 3. Plot Results ---
    print(f"\nPlotting results for include_bias={include_bias}, washout_len={washout_len}, input_scaling={input_scaling}, reset_state_before_predict={reset_state_before_predict}...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_range = range(min(200, len(test_target)))
    ax.plot(test_target[plot_range], "b", label="True Target (with noise)", linewidth=2, alpha=0.7)
    ax.plot(predictions[plot_range], "r--", label="ESN Prediction", linewidth=2)
    ax.set_title(f"ESN: Sine Wave Prediction (Bias={include_bias}, Washout={washout_len}, InputScale={input_scaling}, ResetPredict={reset_state_before_predict})", fontsize=16)
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
    input_scaling = 1.0

    washout_lengths = [0, 50, 100, 200]
    reset_options = [True, False]
    results = {}

    for reset_state_before_predict in reset_options:
        for washout_len in washout_lengths:
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
                input_scaling=input_scaling,
                ridge_alpha=ridge_alpha,
                washout_len=washout_len,
                reset_state_before_predict=reset_state_before_predict,
                plot_filename=f"sine_wave_prediction_bias_true_washout_{washout_len}_reset_{reset_state_before_predict}.png"
            )
            results[(True, washout_len, reset_state_before_predict)] = mse_with_bias

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
                input_scaling=input_scaling,
                ridge_alpha=ridge_alpha,
                washout_len=washout_len,
                reset_state_before_predict=reset_state_before_predict,
                plot_filename=f"sine_wave_prediction_bias_false_washout_{washout_len}_reset_{reset_state_before_predict}.png"
            )
            results[(False, washout_len, reset_state_before_predict)] = mse_without_bias

    print("\n--- Comparison of Results (MSE) ---")
    print("Bias | Washout | ResetPredict | MSE")
    print("-----------------------------------")
    for (bias, washout, reset_predict), mse in results.items():
        print(f"{str(bias):<4} | {washout:<7} | {str(reset_predict):<12} | {mse:.6f}")

    # Suggest further tuning
    print("\n--- Further Tuning Suggestions ---")
    print("If predictions are still poor, consider tuning the following parameters:")
    print("1. spectral_radius: Try values like 0.7, 0.8, 0.9 (must be < 1.0 for stability).")
    print("2. leak_rate: Experiment with 0.1, 0.5, 0.7.")
    print("3. ridge_alpha: Try 1e-6, 1e-3.")
    print("4. input_scaling: Experiment with 0.1, 0.5, 1.0, 2.0.")
    print("5. n_neurons: Increase for more complex tasks, decrease for simpler ones.")


if __name__ == "__main__":
    main()
