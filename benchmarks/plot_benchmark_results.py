#!/usr/bin/env python3
"""Plotting script for benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(csv_file: str) -> None:
    """Read detailed benchmark results from a CSV file, calculate statistics.

    And generate plots for performance and MSE.

    Args:
        csv_file (str): The path to the input CSV file.
    """
    path = Path(csv_file)
    if not path.exists():
        print(f"Error: File not found at '{csv_file}'")
        return

    # Read the raw data
    df = pd.read_csv(csv_file)
    if not isinstance(df, pd.DataFrame):
        print(f"Error: Expected DataFrame from {csv_file}, got {type(df)}")
        return

    # --- Compute 'offline_total' ---
    # Group by threads and run to sum fit and predict times for each unique benchmark run
    offline_grouped = (
        df[df["method"].isin(["offline_fit", "offline_predict"])]
        .groupby(["threads", "run"])["time_s"]
        .sum()
        .reset_index()
    )
    offline_total_df = pd.DataFrame(
        {
            "threads": offline_grouped["threads"],
            "run": offline_grouped["run"],
            "method": "offline_total",
            "time_s": offline_grouped["time_s"],
            "mse": df[df["method"] == "offline_predict"]
            .groupby(["threads", "run"])["mse"]
            .mean()
            .reset_index()["mse"],  # Use predict MSE for total
        }
    )
    df = pd.concat([df, offline_total_df], ignore_index=True)

    # --- 1. Plot Performance (Time vs. Threads) in a single figure ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.lineplot(
        data=df,
        x="threads",
        y="time_s",
        hue="method",
        style="method",
        marker="o",
        markers=True,
        dashes=True,
    )

    ax.set_title("Performance Benchmark: Time vs. Threads", fontsize=16)
    ax.set_xlabel("Number of OpenMP Threads")
    ax.set_ylabel("Average Time (s)")
    ax.legend(title="Method")
    ax.set_yscale("log")  # Use a log scale for the y-axis to better see differences

    # Save the performance plot
    perf_output_path = path.with_name(f"{path.stem}_performance.png")
    plt.savefig(perf_output_path)
    print(f"Performance plot saved to '{perf_output_path}'")

    if sys.stdout.isatty():
        plt.show()

    # --- 2. Plot RLS-only Performance ---
    plt.figure(figsize=(10, 6))
    df_rls = df[df["method"] == "online_rls"]

    ax_rls = sns.lineplot(data=df_rls, x="threads", y="time_s", marker="o")  # type: ignore[reportArgumentType]

    ax_rls.set_title("RLS Performance: Time vs. Threads", fontsize=16)
    ax_rls.set_xlabel("Number of OpenMP Threads")
    ax_rls.set_ylabel("Average Time (s)")

    # Save the RLS performance plot
    rls_output_path = path.with_name(f"{path.stem}_rls_performance.png")
    plt.savefig(rls_output_path)
    print(f"RLS performance plot saved to '{rls_output_path}'")

    if sys.stdout.isatty():
        plt.show()

    # --- 3. Plot MSE ---
    # We only need the MSE for one run, as it should be consistent
    df_mse = df.groupby("method")["mse"].mean().reset_index()

    # Filter out offline_fit and offline_predict
    methods_to_exclude = ["offline_fit", "offline_predict"]
    df_mse_filtered = df_mse[~df_mse["method"].isin(methods_to_exclude)]

    plt.figure(figsize=(10, 6))
    ax_mse = sns.barplot(data=df_mse_filtered, x="method", y="mse")  # type: ignore[reportArgumentType]
    ax_mse.set_title("Comparison of Mean Squared Error (MSE)")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_xlabel("Method")

    # Add MSE values on top of the bars
    for index, row in df_mse_filtered.iterrows():
        ax_mse.text(index, row.mse, f"{row.mse:.4f}", color="black", ha="center")  # type: ignore[reportArgumentType]

    # Save the MSE plot
    mse_output_path = path.with_name(f"{path.stem}_mse.png")
    plt.savefig(mse_output_path)
    print(f"MSE plot saved to '{mse_output_path}'")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from a CSV file.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="benchmarks/benchmark_results.csv",
        help="Path to the CSV file containing benchmark results. Defaults to 'benchmarks/benchmark_results.csv'.",
    )
    args = parser.parse_args()
    plot_results(args.csv_file)
