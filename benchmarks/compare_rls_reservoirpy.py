"""Benchmark comparing rclib and reservoirpy RLS (online learning) performance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import reservoirpy as rpy
from rclib import ESN, readouts, reservoirs
from reservoirpy.nodes import RLS, Reservoir


def mackey_glass(n_samples: int = 5000, tau: int = 17, seed: int = 0) -> np.ndarray:
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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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


def run_benchmarks(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    neuron_sizes: list[int],
    sr: float,
    sparsity: float,
    lr: float,
    input_scaling: float,
    lambda_: float,
) -> pd.DataFrame:
    """Run comparison benchmarks for rclib and reservoirpy RLS."""
    results = []

    print(f"{'Library':<12} | {'Neurons':<8} | {'Online Fit (s)':<15} | {'MSE':<10}")
    print("-" * 55)

    for n in neuron_sizes:
        # rclib
        try:
            res_rc = benchmark_rclib_rls(x_train, y_train, x_test, y_test, n, sr, sparsity, lr, input_scaling, lambda_)
            results.append(res_rc)
            print(
                f"{res_rc['library']:<12} | {res_rc['n_neurons']:<8} | "
                f"{res_rc['online_fit_time']:<15.4f} | {res_rc['mse']:<10.4e}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"rclib failed for n={n}: {e}")

        # reservoirpy
        try:
            res_rpy = benchmark_reservoirpy_rls(x_train, y_train, x_test, y_test, n, sr, sparsity, lr, input_scaling)
            results.append(res_rpy)
            print(
                f"{res_rpy['library']:<12} | {res_rpy['n_neurons']:<8} | "
                f"{res_rpy['online_fit_time']:<15.4f} | {res_rpy['mse']:<10.4e}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"reservoirpy failed for n={n}: {e}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path, plot_suffix: str) -> None:
    """Plot benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="online_fit_time", hue="library", marker="o")
        plt.yscale("log")
        plt.title("RLS Online Learning Time: rclib vs reservoirpy")
        plt.ylabel("Time (s) - Log Scale")
        plt.xlabel("Number of Neurons")
        plt.grid(visible=True, which="both", ls="-", alpha=0.5)
        plot_path = output_dir / f"rls_comparison_time{plot_suffix}"
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("Matplotlib/Seaborn not found. Skipping plot.")


def main() -> None:
    """Run comparison."""
    parser = argparse.ArgumentParser(description="Benchmark comparing rclib and reservoirpy RLS.")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files.")
    parser.add_argument("--plot-suffix", type=str, default=".png", help="Suffix for plot figures (e.g., .png, .pdf).")
    parser.add_argument("--csv-data", type=str, help="Path to existing CSV data to skip benchmarks and only plot.")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.csv_data:
        output_dir = Path(args.csv_data).parent
    else:
        output_dir = Path("benchmarks")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_data:
        print(f"Loading existing data from {args.csv_data}")
        df = pd.read_csv(args.csv_data)
    else:
        # Reduced samples for RLS as it is O(N^2)
        n_samples = 4000
        train_len = 3000
        data = mackey_glass(n_samples=n_samples)

        # Train set
        x_train = data[:train_len]
        y_train = data[1 : train_len + 1]

        # Test set
        x_test = data[train_len:-1]
        y_test = data[train_len + 1 :]

        df = run_benchmarks(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            neuron_sizes=[100, 200, 500, 1000, 1500, 2000, 4000, 8000, 10000],
            sr=0.9,
            sparsity=0.05,
            lr=0.1,
            input_scaling=0.1,
            lambda_=0.99,
        )

        csv_path = output_dir / "rls_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    plot_results(df, output_dir, args.plot_suffix)


if __name__ == "__main__":
    main()
