"Benchmark comparing rclib and reservoirpy performance."

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import reservoirpy as rpy
from rclib import ESN, readouts, reservoirs
from reservoirpy.nodes import Reservoir, Ridge


def mackey_glass(n_samples: int = 5000, tau: int = 17, seed: int = 0) -> np.ndarray:
    """Generate Mackey-Glass time series."""
    rng = np.random.default_rng(seed=seed)
    x = np.zeros(n_samples + tau)
    x[0:tau] = 0.5 + 0.5 * rng.random(tau)
    for t in range(tau, n_samples + tau - 1):
        x[t + 1] = x[t] + (0.2 * x[t - tau]) / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    return x[tau:].reshape(-1, 1)


def benchmark_rclib(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_neurons: int,
    spectral_radius: float,
    sparsity: float,
    leak_rate: float,
    input_scaling: float,
    alpha: float,
    washout: int,
) -> dict[str, Any]:
    """Benchmark rclib."""
    # Setup
    res = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=True,
        seed=42,
    )
    readout = readouts.Ridge(alpha=alpha, include_bias=True)
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # Fit
    start_fit = time.perf_counter()
    model.fit(x_train, y_train, washout_len=washout)
    end_fit = time.perf_counter()

    # Predict
    start_pred = time.perf_counter()
    y_pred = model.predict(x_test, reset_state_before_predict=False)
    end_pred = time.perf_counter()

    # Predict returns x_test.shape[0] predictions
    mse = np.mean((y_pred[:-1] - x_test[1:]) ** 2)

    return {
        "library": "rclib",
        "n_neurons": n_neurons,
        "fit_time": end_fit - start_fit,
        "pred_time": end_pred - start_pred,
        "mse": mse,
    }


def benchmark_reservoirpy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_neurons: int,
    spectral_radius: float,
    sparsity: float,
    leak_rate: float,
    input_scaling: float,
    alpha: float,
    washout: int,
) -> dict[str, Any]:
    """Benchmark reservoirpy."""
    rpy.set_seed(42)

    # reservoirpy 0.4.x params
    reservoir = Reservoir(
        units=n_neurons,
        sr=spectral_radius,
        rc_connectivity=sparsity,
        lr=leak_rate,
        input_scaling=input_scaling,
        # input_bias=True, # Removed, invalid param
        # bias=1.0 # Let's try to enable bias if that's what this does,
        # or we can check docs. For now, let's just match parameters that exist.
        # If rclib has bias and reservoirpy doesn't, comparison is slightly off but
        # acceptable for speed benchmark.
    )

    readout = Ridge(ridge=alpha)
    model = reservoir >> readout

    # Fit
    start_fit = time.perf_counter()
    model.fit(x_train, y_train, warmup=washout)
    end_fit = time.perf_counter()

    # Predict
    start_pred = time.perf_counter()
    y_pred = model.run(x_test)
    end_pred = time.perf_counter()

    mse = np.mean((y_pred[:-1] - x_test[1:]) ** 2)

    return {
        "library": "reservoirpy",
        "n_neurons": n_neurons,
        "fit_time": end_fit - start_fit,
        "pred_time": end_pred - start_pred,
        "mse": mse,
    }


def main() -> None:
    """Run comparison."""
    n_samples = 10000
    train_len = 8000
    washout = 500

    data = mackey_glass(n_samples=n_samples)
    x_train = data[:train_len]
    y_train = data[1 : train_len + 1]
    x_test = data[train_len:-1]

    # Common parameters
    sr = 0.9
    sparsity = 0.05
    lr = 0.1
    input_scaling = 0.1
    alpha = 1e-8

    neuron_sizes = [500, 1000, 2000, 4000, 8000, 10000, 15000, 20000]
    results = []

    print(f"{'Library':<12} | {'Neurons':<8} | {'Fit (s)':<10} | {'Pred (s)':<10} | {'MSE':<10}")
    print("-" * 60)

    for n in neuron_sizes:
        # rclib
        res_rc = benchmark_rclib(x_train, y_train, x_test, n, sr, sparsity, lr, input_scaling, alpha, washout)
        results.append(res_rc)
        print(
            f"{res_rc['library']:<12} | {res_rc['n_neurons']:<8} | {res_rc['fit_time']:<10.4f} | {res_rc['pred_time']:<10.4f} | {res_rc['mse']:<10.4e}"
        )

        # reservoirpy
        try:
            res_rpy = benchmark_reservoirpy(
                x_train, y_train, x_test, n, sr, sparsity, lr, input_scaling, alpha, washout
            )
            results.append(res_rpy)
            print(
                f"{res_rpy['library']:<12} | {res_rpy['n_neurons']:<8} | {res_rpy['fit_time']:<10.4f} | {res_rpy['pred_time']:<10.4f} | {res_rpy['mse']:<10.4e}"
            )
        except Exception as e:
            print(f"reservoirpy failed for n={n}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("benchmarks/comparison_results.csv", index=False)
    print("\nResults saved to benchmarks/comparison_results.csv")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="fit_time", hue="library", marker="o")
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Training Time: rclib vs reservoirpy")
        plt.ylabel("Time (s) - Log Scale")
        plt.xlabel("Number of Neurons - Log Scale")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig("benchmarks/comparison_fit_time.png")

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="pred_time", hue="library", marker="o")
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Prediction Time: rclib vs reservoirpy")
        plt.ylabel("Time (s) - Log Scale")
        plt.xlabel("Number of Neurons - Log Scale")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig("benchmarks/comparison_pred_time.png")
        print("Plots saved as benchmarks/comparison_*.png")
    except ImportError:
        print("Matplotlib/Seaborn not found. Skipping plots.")


if __name__ == "__main__":
    main()
