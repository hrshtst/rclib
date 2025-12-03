# rclib: Reservoir Computing Library

**rclib** is a high-performance, scalable, and general-purpose reservoir computing framework implemented in C++ with Python bindings. It is designed to handle both small-scale networks and very large-scale (40,000+ neurons) architectures, supporting deep (stacked) and parallel reservoir configurations.

## Project Goals

*   **Performance:** Core logic in C++17 using Eigen for linear algebra.
*   **Scalability:** Efficient handling of large sparse reservoirs and complex architectures.
*   **Flexibility:** Modular design separating Reservoirs and Readouts.
*   **Ease of Use:** Pythonic interface via `pybind11` and `scikit-learn` style API.

## Getting Started

### Prerequisites

*   **C++ Compiler:** GCC, Clang, or MSVC supporting C++17.
*   **CMake:** Version 3.15 or higher.
*   **Python:** Version 3.10 or higher (for Python bindings).
*   **Build Tool:** `uv` is recommended for managing the Python environment, but standard `pip` works too.

### Building from Source

1.  **Clone the repository:**
    ```bash
    git clone --recursive https://github.com/hrshtst/rclib.git
    cd rclib
    ```
    *Note: The `--recursive` flag is crucial to fetch dependencies (Eigen, Catch2, pybind11) located in `cpp_core/third_party`.*
    *If you cloned without `--recursive`, run:*
    ```bash
    git submodule update --init --recursive
    ```

2.  **Build C++ Core and Examples:**
    ```bash
    cmake -S . -B build -DBUILD_EXAMPLES=ON
    cmake --build build --config Release -j $(nproc)
    ```

3.  **Run a C++ Example:**
    ```bash
    # Run the Mackey-Glass time series prediction example
    ./build/examples/cpp/mackey_glass
    ```

### Using the Python Interface

You can set up the environment and run Python examples using `uv`:

```bash
# Sync dependencies and run a script
rm -rf .venv && uv sync --no-cache
uv run python examples/python/quick_start.py
```

## Running Tests

### C++ Tests
The project uses `Catch2` for C++ unit testing.

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j $(nproc)
ctest --test-dir build --output-on-failure
```

### Python Tests
The project uses `pytest` for Python integration testing.

```bash
# Ensure the C++ library is built and installed into the python/ directory
cmake -S . -B build
cmake --build build --config Release -j $(nproc) --target _rclib

# Run pytest (via uv)
uv run pytest
```

## Parallelization Configuration

`rclib` provides flexible options to control parallelization strategies, allowing you to optimize for your specific workload and hardware. This is managed via CMake options.

### Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `RCLIB_USE_OPENMP` | `ON` | Enables OpenMP support. Required for any multi-threading. |
| `RCLIB_ENABLE_EIGEN_PARALLELIZATION` | `OFF` | Enables Eigen's internal parallelization (using OpenMP). |

### Recommended Configurations

#### 1. User-Level Parallelism (Default)
**Best for:** Training multiple reservoirs, batch processing, or typical workloads.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=OFF
    ```
*   **How it works:** `rclib` uses OpenMP `#pragma omp parallel for` loops to parallelize high-level operations (e.g., updating state for multiple reservoirs in a parallel architecture, or processing batches). Eigen is forced to run in single-threaded mode to avoid **oversubscription** (too many threads competing for resources).

#### 2. Eigen-Level Parallelism
**Best for:** Very large single networks or dense matrix operations where linear algebra is the bottleneck.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=ON
    ```
*   **How it works:** `rclib` disables its own OpenMP loops. Instead, it lets Eigen use the OpenMP thread pool to parallelize internal matrix operations (like large dense matrix multiplications). This is useful when the reservoir state size is huge.

#### 3. Serial (Single-Threaded)
**Best for:** Debugging or systems without OpenMP.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DRCLIB_USE_OPENMP=OFF
    ```

## Performance Benchmarking

The `benchmarks/` directory contains scripts to evaluate performance across different thread counts and parallelization modes.

1.  **Run the Benchmark Suite:**
    ```bash
    ./benchmarks/benchmark_parallel_comparison.sh
    ```
    This script compiles the project in different modes (Serial, User OMP, Eigen OMP) and runs the `performance_benchmark` executable multiple times.

2.  **Visualize Results:**
    ```bash
    python benchmarks/plot_parallel_comparison.py
    ```
    This generates plots comparing execution time and MSE for different methods and configurations.
