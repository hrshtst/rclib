#!/usr/bin/env python3
"""Visualizes the results from benchmark_parallel_comparison.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MAX_THREADS_FOR_TICKS = 32


def plot_metric(data: pd.DataFrame, title: str, filename_suffix: str, path: Path) -> None:
    """Plot a specific metric comparing modes."""
    plt.figure(figsize=(10, 6))

    # Plot parallel modes as lines
    parallel_modes = ["user_omp", "eigen_omp"]
    parallel_data = data[data["mode"].isin(parallel_modes)]

    if not parallel_data.empty:
        sns.lineplot(
            data=parallel_data,  # type: ignore[reportArgumentType]
            x="threads",
            y="time_s",
            hue="mode",
            hue_order=parallel_modes,
            style="mode",
            markers=True,
            dashes=False,
            linewidth=2.5,
            alpha=0.8,
        )

    # Plot serial baseline as a horizontal line
    serial_data = data[data["mode"] == "serial"]
    if not serial_data.empty:
        serial_avg = float(serial_data["time_s"].mean())
        plt.axhline(y=serial_avg, color="r", linestyle="--", label=f"Serial Baseline ({serial_avg:.4f}s)")

    plt.title(title, fontsize=16)
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (s)")
    plt.legend(title="Parallelization Mode")
    plt.yscale("log")

    # Force integer ticks on x-axis if not too many
    max_threads = data["threads"].max()
    if max_threads <= MAX_THREADS_FOR_TICKS:
        plt.xticks(data["threads"].unique().tolist())

    output_path = path.with_name(f"{path.stem}_{filename_suffix}.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    if sys.stdout.isatty():
        plt.show()
    plt.close()


def plot_mode_methods(df: pd.DataFrame, mode_name: str, nice_name: str, filename_suffix: str, path: Path) -> None:
    """Plot Method Comparison (Offline vs RLS vs LMS) for a specific mode."""
    subset = df[df["mode"] == mode_name]
    if subset.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset,  # type: ignore[reportArgumentType]
        x="threads",
        y="time_s",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        linewidth=2.5,
        alpha=0.8,
    )

    plt.title(f"Method Performance Comparison ({nice_name})", fontsize=16)
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (s)")
    plt.legend(title="Method")
    plt.yscale("log")

    # Force integer ticks on x-axis if not too many
    max_threads = subset["threads"].max()
    if max_threads <= MAX_THREADS_FOR_TICKS:
        plt.xticks(subset["threads"].unique().tolist())

    output_path = path.with_name(f"{path.stem}_{filename_suffix}.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    if sys.stdout.isatty():
        plt.show()
    plt.close()


def plot_mse(df: pd.DataFrame, path: Path) -> None:
    """Plot MSE Comparison."""
    # Filter for relevant modes
    subset = df[df["mode"].isin(["user_omp", "eigen_omp"])]
    if subset.empty:
        return

    # Aggregate to get mean MSE across threads/runs (MSE shouldn't vary by thread count)
    mse_agg = subset.groupby(["mode", "method"])["mse"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=mse_agg, x="method", y="mse", hue="mode", hue_order=["user_omp", "eigen_omp"])

    plt.title("MSE Comparison: Offline vs LMS vs RLS", fontsize=16)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xlabel("Method")
    plt.legend(title="Parallelization Mode")

    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2e", padding=3)  # type: ignore[reportArgumentType]

    output_path = path.with_name(f"{path.stem}_mse_comparison.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    if sys.stdout.isatty():
        plt.show()
    plt.close()


def plot_online_mse(df: pd.DataFrame, path: Path) -> None:
    """Plot Focused Online MSE Comparison."""
    # Filter for relevant modes and ONLY online methods
    subset = df[df["mode"].isin(["user_omp", "eigen_omp"])]
    subset = subset[subset["method"].str.contains("Online")]
    if subset.empty:
        return

    # Aggregate to get mean MSE across threads/runs
    mse_agg = subset.groupby(["mode", "method"])["mse"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=mse_agg, x="method", y="mse", hue="mode", hue_order=["user_omp", "eigen_omp"])

    plt.title("Online Learning MSE Comparison: LMS vs RLS", fontsize=16)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xlabel("Method")
    plt.legend(title="Parallelization Mode")

    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2e", padding=3)  # type: ignore[reportArgumentType]

    output_path = path.with_name(f"{path.stem}_online_mse_comparison.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    if sys.stdout.isatty():
        plt.show()
    plt.close()


def plot_comparison(csv_file: str) -> None:
    """Visualize the results from benchmark_parallel_comparison.sh."""
    path = Path(csv_file)
    if not path.exists():
        print(f"Error: File not found at '{csv_file}'")
        return

    # Load Data
    df = pd.read_csv(csv_file)

    # --- Preprocessing ---

    # 1. Calculate Total Offline Time (Fit + Predict) per run
    offline_data = df[df["method"].isin(["offline_fit", "offline_predict"])]
    # Sum time_s
    offline_time = offline_data.groupby(["mode", "threads", "run"])["time_s"].sum().reset_index()
    # Get MSE from predict (it should be the same for fit, or at least predict is what matters)
    offline_mse = df[df["method"] == "offline_predict"].groupby(["mode", "threads", "run"])["mse"].mean().reset_index()

    offline_total = offline_time.merge(offline_mse, on=["mode", "threads", "run"])
    offline_total["method"] = "Offline Total (Fit + Predict)"

    # 2. Select Online Methods
    online_lms = df[df["method"] == "online_lms"].copy()
    online_lms["method"] = "Online LMS"

    online_rls = df[df["method"] == "online_rls"].copy()
    online_rls["method"] = "Online RLS"

    # Combine all for easier plotting later
    combined_df = pd.concat([offline_total, online_lms, online_rls], ignore_index=True)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")

    # Plot 1: Offline Performance
    plot_metric(cast("pd.DataFrame", offline_total), "Batch Training & Prediction Performance", "offline", path)

    # Plot 2: Online RLS Performance
    plot_metric(cast("pd.DataFrame", online_rls), "Online RLS Performance", "rls", path)

    # Plot 3: Online LMS Performance
    plot_metric(cast("pd.DataFrame", online_lms), "Online LMS Performance", "lms", path)

    # Plot 4: Method Comparison for User OMP
    plot_mode_methods(cast("pd.DataFrame", combined_df), "user_omp", "User Parallelism", "methods_user_omp", path)

    # Plot 5: Method Comparison for Eigen OMP
    plot_mode_methods(cast("pd.DataFrame", combined_df), "eigen_omp", "Eigen Parallelism", "methods_eigen_omp", path)

    # Plot 6: MSE Comparison
    plot_mse(cast("pd.DataFrame", combined_df), path)

    # Plot 7: Focused Online MSE Comparison
    plot_online_mse(cast("pd.DataFrame", combined_df), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parallel benchmark results.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="benchmarks/parallel_comparison_results.csv",
        help="Path to the CSV results file.",
    )
    args = parser.parse_args()
    plot_comparison(args.csv_file)
