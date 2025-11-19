#!/bin/bash

# A script to benchmark the performance of an executable with different
# numbers of OpenMP threads.

# --- Configuration ---
EXECUTABLE=$1
OUTPUT_FILE=${2:-benchmarks/benchmark_results.csv} # Default output file
NUM_RUNS=10
THREAD_COUNTS=(1 2 4 8 12 16 24 32)

# --- Validation ---
if [ -z "$EXECUTABLE" ]; then
    echo "Usage: $0 <path_to_executable> [output_csv_file]"
    exit 1
fi

if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or is not executable."
    exit 1
fi

# --- Benchmark ---
echo "--- OpenMP Thread Benchmark ---"
echo "Executable: $EXECUTABLE"
echo "Number of runs per test: $NUM_RUNS"
echo "Saving results to: $OUTPUT_FILE"
echo "---------------------------------"
printf "% -15s | % -s\n" "OMP_NUM_THREADS" "Status"
echo "---------------------------------"

# Create/clear the output file and write the header
echo "threads,run,method,time_s,mse" > "$OUTPUT_FILE"

for threads in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$threads
    printf "% -15s | % -s" "$threads" "Running..."

    # Run the executable multiple times and record each result
    for (( i=1; i<=NUM_RUNS; i++ )); do
        # Run the C++ benchmark program and capture its CSV output, skipping the header
        # Use awk to prepend the thread count and run number
        "$EXECUTABLE" | tail -n +2 | awk -v threads="$threads" -v run="$i" -F, '{print threads","run","$0}' >> "$OUTPUT_FILE"
    done

    printf "\r% -15s | % -s\n" "$threads" "Done      "

done

echo "---------------------------------"
echo "Benchmark complete. Results saved in $OUTPUT_FILE"
