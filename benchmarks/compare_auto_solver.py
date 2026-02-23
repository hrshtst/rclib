"""Benchmark comparing rclib (auto solver) and reservoirpy performance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
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


def benchmark_rclib_auto(
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
    """Benchmark rclib with auto solver."""
    res = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        include_bias=True,
        seed=42,
    )
    readout = readouts.Ridge(alpha=alpha, include_bias=True, solver="auto")
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    start_fit = time.perf_counter()
    model.fit(x_train, y_train, washout_len=washout)
    end_fit = time.perf_counter()

    effective_solver = model._cpp_model.getReadout().getEffectiveSolver().name  # noqa: SLF001

    start_pred = time.perf_counter()
    y_pred = model.predict(x_test, reset_state_before_predict=False)
    end_pred = time.perf_counter()

    mse = np.mean((y_pred[:-1] - x_test[1:]) ** 2)

    return {
        "library": f"rclib (auto:{effective_solver.lower()})",
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

    reservoir = Reservoir(
        units=n_neurons,
        sr=spectral_radius,
        rc_connectivity=sparsity,
        lr=leak_rate,
        input_scaling=input_scaling,
    )

    readout = Ridge(ridge=alpha)
    model = reservoir >> readout

    start_fit = time.perf_counter()
    model.fit(x_train, y_train, warmup=washout)
    end_fit = time.perf_counter()

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


def run_benchmarks(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    neuron_sizes: list[int],
    sr: float,
    sparsity: float,
    lr: float,
    input_scaling: float,
    alpha: float,
    washout: int,
) -> pd.DataFrame:
    """Run comparison benchmarks for rclib (auto) and reservoirpy."""
    results = []

    print(f"{'Library':<40} | {'Neurons':<8} | {'Fit (s)':<10} | {'Pred (s)':<10} | {'MSE':<10}")
    print("-" * 90)

    for n in neuron_sizes:
        res_rc = benchmark_rclib_auto(x_train, y_train, x_test, n, sr, sparsity, lr, input_scaling, alpha, washout)
        results.append(res_rc)
        print(
            f"{res_rc['library']:<40} | {res_rc['n_neurons']:<8} | "
            f"{res_rc['fit_time']:<10.4f} | {res_rc['pred_time']:<10.4f} | {res_rc['mse']:<10.4e}"
        )

        try:
            res_rpy = benchmark_reservoirpy(
                x_train, y_train, x_test, n, sr, sparsity, lr, input_scaling, alpha, washout
            )
            results.append(res_rpy)
            print(
                f"{res_rpy['library']:<40} | {res_rpy['n_neurons']:<8} | "
                f"{res_rpy['fit_time']:<10.4f} | {res_rpy['pred_time']:<10.4f} | {res_rpy['mse']:<10.4e}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"reservoirpy failed for n={n}: {e}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path, plot_suffix: str) -> None:
    """Plot benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        title_fs = 18
        label_fs = 16
        tick_fs = 14
        legend_fs = 14

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="fit_time", hue="library", marker="o")
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Training Time: rclib (auto) vs reservoirpy", fontsize=title_fs)
        plt.ylabel("Time (s) - Log Scale", fontsize=label_fs)
        plt.xlabel("Number of Neurons - Log Scale", fontsize=label_fs)
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.legend(fontsize=legend_fs)
        plt.grid(visible=True, which="both", ls="-", alpha=0.5)
        fit_plot_path = output_dir / f"comparison_auto_fit_time{plot_suffix}"
        plt.tight_layout()
        plt.savefig(fit_plot_path)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="n_neurons", y="pred_time", hue="library", marker="o")
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Prediction Time: rclib (auto) vs reservoirpy", fontsize=title_fs)
        plt.ylabel("Time (s) - Log Scale", fontsize=label_fs)
        plt.xlabel("Number of Neurons - Log Scale", fontsize=label_fs)
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.legend(fontsize=legend_fs)
        plt.grid(visible=True, which="both", ls="-", alpha=0.5)
        pred_plot_path = output_dir / f"comparison_auto_pred_time{plot_suffix}"
        plt.tight_layout()
        plt.savefig(pred_plot_path)
        print(f"Plots saved to {output_dir}/comparison_auto_*")
    except ImportError:
        print("Matplotlib/Seaborn not found. Skipping plots.")


def main() -> None:
    """Run comparison."""
    parser = argparse.ArgumentParser(description="Benchmark comparing rclib (auto) and reservoirpy.")
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
        n_samples = 10000
        train_len = 8000
        washout = 500

        data = mackey_glass(n_samples=n_samples)
        x_train = data[:train_len]
        y_train = data[1 : train_len + 1]
        x_test = data[train_len:-1]

        df = run_benchmarks(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            neuron_sizes=[500, 1000, 2000, 4000, 8000, 10000, 15000, 20000],
            sr=0.9,
            sparsity=0.05,
            lr=0.1,
            input_scaling=0.1,
            alpha=1e-8,
            washout=washout,
        )

        csv_path = output_dir / "comparison_auto_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    plot_results(df, output_dir, args.plot_suffix)


if __name__ == "__main__":
    main()
