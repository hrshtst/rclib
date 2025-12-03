import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison(csv_file: str):
    """
    Visualizes the results from benchmark_parallel_comparison.sh
    """
    if not os.path.exists(csv_file):
        print(f"Error: File not found at '{csv_file}'")
        return

    # Load Data
    df = pd.read_csv(csv_file)

    # --- Preprocessing ---
    
    # 1. Calculate Total Offline Time (Fit + Predict) per run
    offline_data = df[df['method'].isin(['offline_fit', 'offline_predict'])]
    offline_total = offline_data.groupby(['mode', 'threads', 'run'])['time_s'].sum().reset_index()
    offline_total['method'] = 'Offline Total (Fit + Predict)'
    
    # 2. Select Online Methods
    online_lms = df[df['method'] == 'online_lms'].copy()
    online_lms['method'] = 'Online LMS'
    
    online_rls = df[df['method'] == 'online_rls'].copy()
    online_rls['method'] = 'Online RLS'

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    
    # Helper to plot a specific metric
    def plot_metric(data, title, filename_suffix):
        plt.figure(figsize=(10, 6))
        
        # Plot parallel modes as lines
        parallel_modes = ['user_omp', 'eigen_omp']
        parallel_data = data[data['mode'].isin(parallel_modes)]
        
        if not parallel_data.empty:
            sns.lineplot(
                data=parallel_data,
                x="threads",
                y="time_s",
                hue="mode",
                style="mode",
                markers=True,
                dashes=False,
                linewidth=2.5,
                alpha=0.8
            )

        # Plot serial baseline as a horizontal line
        serial_data = data[data['mode'] == 'serial']
        if not serial_data.empty:
            serial_avg = serial_data['time_s'].mean()
            plt.axhline(y=serial_avg, color='r', linestyle='--', label=f'Serial Baseline ({serial_avg:.4f}s)')

        plt.title(title, fontsize=16)
        plt.xlabel("Number of Threads")
        plt.ylabel("Time (s)")
        plt.legend(title="Parallelization Mode")
        plt.yscale('log')
        
        # Force integer ticks on x-axis if not too many
        max_threads = data['threads'].max()
        if max_threads <= 32:
             plt.xticks(data['threads'].unique())

        output_file = os.path.splitext(csv_file)[0] + f'_{filename_suffix}.png'
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

    # Plot 1: Offline Performance
    plot_metric(offline_total, "Batch Training & Prediction Performance", "offline")

    # Plot 2: Online RLS Performance
    # RLS involves matrix operations that *might* benefit from Eigen parallelism
    plot_metric(online_rls, "Online RLS Performance", "rls")

    # Plot 3: Online LMS Performance
    # LMS is vector-vector, less likely to scale with threads compared to overhead, but good to check
    plot_metric(online_lms, "Online LMS Performance", "lms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parallel benchmark results.")
    parser.add_argument(
        "csv_file",
        nargs='?',
        default="benchmarks/parallel_comparison_results.csv",
        help="Path to the CSV results file."
    )
    args = parser.parse_args()
    plot_comparison(args.csv_file)
