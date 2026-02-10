"""Benchmark script to sweep through different tolerance values for Ridge CG solvers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from rclib import ESN, readouts, reservoirs


def mackey_glass(n_samples: int = 5000, tau: int = 17, seed: int = 0) -> np.ndarray:
    """Generate Mackey-Glass time series.

    Args:
        n_samples: Number of samples to generate.
        tau: Delay parameter.
        seed: Random seed.

    Returns
    -------
        Generated time series as a numpy array.
    """
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
    alpha: float,
    washout: int,
    solver: str,
    tolerance: float,
) -> dict[str, Any]:
    """Benchmark rclib with a specific solver and tolerance.

    Args:
        x_train: Training input data.
        y_train: Training target data.
        x_test: Test input data.
        n_neurons: Number of neurons in the reservoir.
        alpha: Regularization parameter.
        washout: Number of washout samples.
        solver: Solver name.
        tolerance: Convergence tolerance for iterative solvers.

    Returns
    -------
        A dictionary containing the benchmark results.
    """
    res = reservoirs.RandomSparse(
        n_neurons=n_neurons,
        spectral_radius=0.9,
        sparsity=0.05,
        leak_rate=0.1,
        input_scaling=0.1,
        include_bias=True,
        seed=42,
    )
    readout = readouts.Ridge(alpha=alpha, include_bias=True, solver=solver, tolerance=tolerance)
    model = ESN()
    model.add_reservoir(res)
    model.set_readout(readout)

    start_fit = time.perf_counter()
    model.fit(x_train, y_train, washout_len=washout)
    end_fit = time.perf_counter()

    y_pred = model.predict(x_test, reset_state_before_predict=False)
    mse = np.mean((y_pred[:-1] - x_test[1:]) ** 2)

    return {"solver": solver, "tolerance": tolerance, "fit_time": end_fit - start_fit, "mse": mse}


def main() -> None:
    """Run the tolerance sweep benchmark."""
    n_samples = 10000
    train_len = 8000
    washout = 500
    data = mackey_glass(n_samples=n_samples)
    x_train = data[:train_len]
    y_train = data[1 : train_len + 1]
    x_test = data[train_len:-1]

    n_neurons = 1000
    alpha = 1e-8
    tolerances = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    solvers = ["conjugate_gradient", "conjugate_gradient_implicit"]

    # Baseline Cholesky
    res_ch = benchmark_rclib(x_train, y_train, x_test, n_neurons, alpha, washout, "cholesky", 0.0)
    print(f"Cholesky Baseline - Fit: {res_ch['fit_time']:.4f}s, MSE: {res_ch['mse']:.4e}")

    print(f"{'Solver':<30} | {'Tolerance':<10} | {'Fit (s)':<10} | {'MSE':<10}")
    print("-" * 65)

    for solver in solvers:
        for tol in tolerances:
            res = benchmark_rclib(x_train, y_train, x_test, n_neurons, alpha, washout, solver, tol)
            print(f"{solver:<30} | {tol:<10.0e} | {res['fit_time']:<10.4f} | {res['mse']:<10.4e}")


if __name__ == "__main__":
    main()
