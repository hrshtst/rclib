import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_file: str):
    """
    Reads benchmark results from a CSV file, calculates statistics,
    and generates a plot with error bars.

    Args:
        csv_file (str): The path to the input CSV file.
    """
    if not os.path.exists(csv_file):
        print(f"Error: File not found at '{csv_file}'")
        return

    # Read the raw data
    df_raw = pd.read_csv(csv_file)

    # Calculate mean and standard deviation for each thread count
    df_agg = df_raw.groupby('threads')['time_s'].agg(['mean', 'std']).reset_index()
    df_agg.rename(columns={'mean': 'avg_time_s', 'std': 'std_time_s'}, inplace=True)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Create a bar plot for the average time
    ax = sns.barplot(x='threads', y='avg_time_s', data=df_agg, color='b', alpha=0.6, label='Execution Time')
    
    # Add error bars
    ax.errorbar(
        x=ax.get_xticks(),
        y=df_agg['avg_time_s'],
        yerr=df_agg['std_time_s'],
        fmt='none',
        capsize=5,
        color='black',
        label='Std Dev'
    )

    ax.set_ylabel('Average Time (s)')
    ax.set_xlabel('Number of OpenMP Threads')
    ax.set_title('Benchmark: Execution Time vs. Number of Threads')

    # Calculate and plot speedup on a secondary y-axis
    baseline_time = df_agg[df_agg['threads'] == 1]['avg_time_s'].iloc[0]
    df_agg['speedup'] = baseline_time / df_agg['avg_time_s']
    
    ax2 = ax.twinx()
    sns.lineplot(x='threads', y='speedup', data=df_agg, marker='o', color='r', ax=ax2, label='Speedup')
    ax2.set_ylabel('Speedup (relative to 1 thread)')
    ax2.grid(False)

    # --- Final Touches ---
    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Manually add the error bar to the legend
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='black', lw=1, label='Std Dev'))
    labels.append('Std Dev')
    ax2.legend(handles=handles + handles2, labels=labels + labels2, loc='upper left')
    
    # Save the plot
    output_filename = os.path.splitext(csv_file)[0] + '.png'
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from a CSV file.")
    parser.add_argument(
        "csv_file",
        nargs='?',
        default="benchmarks/benchmark_results.csv",
        help="Path to the CSV file containing benchmark results. Defaults to 'benchmarks/benchmark_results.csv'."
    )
    args = parser.parse_args()
    plot_results(args.csv_file)