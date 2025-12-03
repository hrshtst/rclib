#!/bin/bash

# Script to benchmark performance across three parallelization modes:
# 1. Serial (Baseline)
# 2. User Parallelism (OpenMP explicit loops)
# 3. Eigen Parallelism (Internal threading)

# --- Configuration ---
OUTPUT_FILE=${1:-benchmarks/parallel_comparison_results.csv}
NUM_RUNS=${2:-10}
THREAD_COUNTS=(1 2 4 8 12 16 24 32)

# Ensure we are in the project root
cd "$(dirname "$0")/.." || exit 1

# Output Header
echo "mode,threads,run,method,time_s,mse" > "$OUTPUT_FILE"

# Function to parse and append results
# Arguments: mode, threads, run_number, executable_path
run_and_record() {
    local mode=$1
    local threads=$2
    local run_num=$3
    local exe=$4

    # Run executable, skip header line, parse CSV output
    "$exe" | tail -n +2 | while IFS=, read -r method time_s mse; do
        echo "$mode,$threads,$run_num,$method,$time_s,$mse" >> "$OUTPUT_FILE"
    done
}

echo "Starting Benchmark Suite (Runs per config: $NUM_RUNS)"
echo "Results will be saved to: $OUTPUT_FILE"
echo "--------------------------------------------------"

# --- 1. Serial Benchmark ---
echo "[1/3] Running Serial Benchmark (RCLIB_USE_OPENMP=OFF)..."
cmake -S . -B build_serial -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=OFF -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=OFF > /dev/null
cmake --build build_serial --target performance_benchmark --config Release > /dev/null

for (( i=1; i<=NUM_RUNS; i++ )); do
    run_and_record "serial" 1 "$i" "build_serial/examples/cpp/performance_benchmark"
done
echo "      Done."

# --- 2. User Parallelism (OpenMP) ---
echo "[2/3] Running User Parallelism Benchmark (RCLIB_USE_OPENMP=ON, RCLIB_ENABLE_EIGEN_PARALLELIZATION=OFF)..."
cmake -S . -B build_user_omp -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=OFF > /dev/null
cmake --build build_user_omp --target performance_benchmark --config Release > /dev/null

for threads in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$threads
    printf "      Threads: %-2d | " "$threads"
    for (( i=1; i<=NUM_RUNS; i++ )); do
        run_and_record "user_omp" "$threads" "$i" "build_user_omp/examples/cpp/performance_benchmark"
        printf "."
    done
    printf "\n"
done

# --- 3. Eigen Parallelism ---
echo "[3/3] Running Eigen Parallelism Benchmark (RCLIB_USE_OPENMP=ON, RCLIB_ENABLE_EIGEN_PARALLELIZATION=ON)..."
cmake -S . -B build_eigen_omp -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=ON > /dev/null
cmake --build build_eigen_omp --target performance_benchmark --config Release > /dev/null

for threads in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$threads
    printf "      Threads: %-2d | " "$threads"
    for (( i=1; i<=NUM_RUNS; i++ )); do
        run_and_record "eigen_omp" "$threads" "$i" "build_eigen_omp/examples/cpp/performance_benchmark"
        printf "."
    done
    printf "\n"
done

echo "--------------------------------------------------"
echo "Benchmark Suite Complete."
