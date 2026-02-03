"""Benchmark comparing rclib and reservoirpy RLS (online learning) performance."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import reservoirpy as rpy
from rclib import ESN, readouts, reservoirs
from reservoirpy.nodes import RLS, Reservoir


def mackey_glass(n_samples: int = 5000, tau: int = 17, seed: int = 0):
    """Generate Mackey-Glass time series."""
    rng = np.random.default_rng(seed=seed)
    x = np.zeros(n_samples + tau)
    x[0:tau] = 0.5 + 0.5 * rng.random(tau)
    for t in range(tau, n_samples + tau - 1):
        x[t + 1] = x[t] + (0.2 * x[t - tau]) / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    return x[tau:].reshape(-1, 1)


def benchmark_rclib_rls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_neurons: int,
    spectral_radius: float,
    sparsity: float,
    leak_rate: float,
    input_scaling: float,
    lambda_: float,
):
    """Benchmark rclib RLS."""
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
    # RLS Readout
    readout = readouts.Rls(lambda_=lambda_, delta=1.0, include_bias=True)
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    # Online Fit (using partial_fit in a loop)
    start_time = time.perf_counter()
    for i in range(len(x_train)):
        model.partial_fit(x_train[i : i + 1], y_train[i : i + 1])
    end_time = time.perf_counter()

    # Predict on test set
    y_pred = model.predict(x_test, reset_state_before_predict=False)
    mse = np.mean((y_pred - y_test) ** 2)

    return {
        "library": "rclib",
        "n_neurons": n_neurons,
        "online_fit_time": end_time - start_time,
        "mse": mse,
    }


def benchmark_reservoirpy_rls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_neurons: int,
    spectral_radius: float,
    sparsity: float,
    leak_rate: float,
    input_scaling: float,
):
    """Benchmark reservoirpy RLS (FORCE)."""
    rpy.set_seed(42)

    reservoir = Reservoir(
        units=n_neurons,
        sr=spectral_radius,
        rc_connectivity=sparsity,
        lr=leak_rate,
        input_scaling=input_scaling,
    )
    # RLS is the node in reservoirpy
    readout = RLS()
    model = reservoir >> readout

    # Online Fit
    start_time = time.perf_counter()
    # fit() with an RLS node performs online updates
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict
    y_pred = model.run(x_test)
    mse = np.mean((y_pred - y_test) ** 2)

    return {
        "library": "reservoirpy",
        "n_neurons": n_neurons,
        "online_fit_time": end_time - start_time,
        "mse": mse,
    }


def main() -> None:
    """Run comparison."""
    # Reduced samples for RLS as it is O(N^2)
    n_samples = 4000
    train_len = 3000
    data = mackey_glass(n_samples=n_samples)
    
    # Train set
    x_train = data[:train_len]
    y_train = data[1 : train_len + 1]
    
    # Test set
    x_test = data[train_len:-1]
    y_test = data[train_len + 1:]

    # Common parameters
    sr = 0.9
    sparsity = 0.05
    lr = 0.1
    input_scaling = 0.1
    lambda_ = 0.99  # Forgetting factor

    neuron_sizes = [100, 200, 500, 1000, 1500]
    results = []

    print(f"{ 'Library':<12} | { 'Neurons':<8} | { 'Online Fit (s)':<15} | { 'MSE':<10}")
    print("-" * 55)

    for n in neuron_sizes:
        # rclib
        try:
            res_rc = benchmark_rclib_rls(x_train, y_train, x_test, y_test, n, sr, sparsity, lr, input_scaling, lambda_)
            results.append(res_rc)
            print(f"{res_rc['library']:<12} | {res_rc['n_neurons']:<8} | {res_rc['online_fit_time']:<15.4f} | {res_rc['mse']:<10.4e}")
        except Exception as e:
            print(f"rclib failed for n={n}: {e}")

        # reservoirpy
        try:
            res_rpy = benchmark_reservoirpy_rls(x_train, y_train, x_test, y_test, n, sr, sparsity, lr, input_scaling)
            results.append(res_rpy)
            print(f"{res_rpy['library']:<12} | {res_rpy['n_neurons']:<8} | {res_rpy['online_fit_time']:<15.4f} | {res_rpy['mse']:<10.4e}")
        except Exception as e:
            print(f"reservoirpy failed for n={n}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("benchmarks/rls_comparison_results.csv", index=False)
    print("\nResults saved to benchmarks/rls_comparison_results.csv")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="online_fit_time", hue="library", marker="o")
        plt.yscale("log")
        plt.title("RLS Online Learning Time: rclib vs reservoirpy")
        plt.ylabel("Time (s) - Log Scale")
        plt.xlabel("Number of Neurons")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig("benchmarks/rls_comparison_time.png")
        print("Plot saved as benchmarks/rls_comparison_time.png")
    except ImportError:
        print("Matplotlib/Seaborn not found. Skipping plot.")


if __name__ == "__main__":
    main()
