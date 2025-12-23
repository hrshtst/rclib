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
*   **OpenMP:** Required for parallelization.
    *   Ubuntu/Debian: `sudo apt install libomp-dev`

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
    cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j $(nproc)
    ```

3.  **Run a C++ Example:**
    ```bash
    # Run the Mackey-Glass time series prediction example
    ./build/examples/cpp/mackey_glass
    ```

### Using the Python Interface

This project provides Python bindings for the core C++ code, leveraging `uv`, `scikit-build-core`, and `pybind11`.

To enable fast incremental builds and automatic rebuilding when C++ source files change (see [astral-sh/uv#13998](https://github.com/astral-sh/uv/issues/13998)), use the following two-step installation process:

```bash
# 1. Install build dependencies without installing the project
uv sync --no-install-project --only-group build

# 2. Install the project and remaining dependencies
uv sync

# Run the quick start example
uv run python examples/python/quick_start.py
```

With this configuration, any changes to the C++ source code in `cpp_core` will automatically trigger a rebuild of the Python extension module upon the next import, ensuring your Python environment always uses the latest C++ logic without manual recompilation.

### Integrating `rclib_core` into Your C++ Project (CMake)

If you wish to use `rclib_core` as a static library in your own C++ project, the recommended approach is to add it as a Git submodule.

1.  **Add `rclib` as a Git Submodule:**
    Navigate to your project's root directory and add `rclib` as a submodule:
    ```bash
    git submodule add https://github.com/hrshtst/rclib.git third_party/rclib
    git submodule update --init --recursive third_party/rclib
    ```
    (Adjust `third_party/rclib` to your desired path.)

2.  **Integrate with Your `CMakeLists.txt`:**
    In your project's `CMakeLists.txt` file, add `rclib` as a subdirectory and link against its `rclib_core` target. Ensure you propagate relevant build options like OpenMP if needed.

    ```cmake
    # Add rclib as a subdirectory
    add_subdirectory(third_party/rclib)

    # Example of how to link rclib_core to your target executable or library
    add_executable(my_rc_app main.cpp)
    target_link_libraries(my_rc_app PRIVATE rclib_core)

    # Note: rclib_core internally handles its Eigen dependency.
    # If your project directly uses Eigen, ensure it's properly configured in your CMakeLists.txt.
    ```

3.  **Configure Parallelization (Optional):**
    If your project also uses OpenMP or needs to control `rclib`'s parallelization, you can set the `RCLIB_USE_OPENMP` and `RCLIB_ENABLE_EIGEN_PARALLELIZATION` CMake options *before* calling `add_subdirectory(third_party/rclib)`.

    ```cmake
    set(RCLIB_USE_OPENMP ON CACHE BOOL "Enable OpenMP support in rclib_core")
    set(RCLIB_ENABLE_EIGEN_PARALLELIZATION OFF CACHE BOOL "Enable Eigen's internal parallelization in rclib_core")
    add_subdirectory(third_party/rclib)
    # ... rest of your project's CMakeLists.txt
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
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=OFF
    ```
*   **How it works:** `rclib` uses OpenMP `#pragma omp parallel for` loops to parallelize high-level operations (e.g., updating state for multiple reservoirs in a parallel architecture, or processing batches). Eigen is forced to run in single-threaded mode to avoid **oversubscription** (too many threads competing for resources).

#### 2. Eigen-Level Parallelism
**Best for:** Very large single networks or dense matrix operations where linear algebra is the bottleneck.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=ON
    ```
*   **How it works:** `rclib` disables its own OpenMP loops. Instead, it lets Eigen use the OpenMP thread pool to parallelize internal matrix operations (like large dense matrix multiplications). This is useful when the reservoir state size is huge.

#### 3. Serial (Single-Threaded)
**Best for:** Debugging or systems without OpenMP.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRCLIB_USE_OPENMP=OFF
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
    uv run python benchmarks/plot_parallel_comparison.py
    ```
    This generates plots comparing execution time and MSE for different methods and configurations.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
