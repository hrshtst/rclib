# rclib: Reservoir Computing Library

**rclib** is a high-performance, scalable, and general-purpose reservoir computing framework implemented in C++ with Python bindings. It is designed to handle both small-scale networks and very large-scale (40,000+ neurons) architectures, supporting deep (stacked) and parallel reservoir configurations.

## Project Goals

*   **Performance:** Core logic in C++17 using Eigen for linear algebra.
*   **Scalability:** Efficient handling of large sparse reservoirs and complex architectures.
*   **Flexibility:** Modular design separating Reservoirs and Readouts.
*   **Ease of Use:** Pythonic interface via `pybind11` and `scikit-learn` style API.
*   **Reproducibility:** Deterministic results via explicit seeding of random reservoirs.

## Current Directory Structure

```
rclib/
├── benchmarks/            # Performance benchmarking scripts
├── build/                 # Build directory (default)
├── cpp_core/              # C++ source
│   ├── include/rclib/     # Public headers
│   ├── src/               # Source files
│   └── third_party/       # Dependencies (Eigen, Catch2, pybind11)
├── docs/                  # Documentation source (mkdocs)
├── examples/              # Examples
│   ├── cpp/
│   └── python/
├── python/                # Python package structure
│   └── rclib/
├── scripts/               # Helper scripts (e.g., version bumping)
├── tests/                 # Unit and integration tests
│   ├── cpp/
│   └── python/
├── CMakeLists.txt         # Main CMake build file
├── LICENSE                # Apache License 2.0
├── README.md
├── noxfile.py             # Automation configuration
├── mkdocs.yml             # Documentation configuration
├── pyproject.toml         # Python project configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .clang-format          # C++ formatting configuration
├── .cmake-format.yaml     # CMake formatting configuration
├── .geminiignore          # Files ignored by Gemini
└── GEMINI.md              # This context file
```

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

2.  **Build C++ Core and Examples:**
    ```bash
    # Build with examples enabled (defaults: Release type, Export Compile Commands ON)
    cmake -S . -B build -DBUILD_EXAMPLES=ON
    cmake --build build --config Release -j $(nproc)
    ```

3.  **Run a C++ Example:**
    ```bash
    # Run the Mackey-Glass time series prediction example (if built with -DBUILD_EXAMPLES=ON)
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

## Development Workflow

### Code Quality Tools

The project uses several tools to ensure code quality:

*   **Ruff:** For Python linting and formatting.
*   **Basedpyright:** For static type checking.
*   **clang-format:** For C++ formatting (LLVM style).
*   **shellcheck:** For shell script linting.
*   **cmake-format / cmake-lint:** For CMake formatting and linting.
*   **Pre-commit:** To enforce checks before committing.

### Setting up Pre-commit

```bash
uv run pre-commit install
```

### Running Tests

#### C++ Tests
The project uses `Catch2` for C++ unit testing.

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j $(nproc)
ctest --test-dir build --output-on-failure
```

#### Python Tests
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

#### 2. Eigen-Level Parallelism
**Best for:** Very large single networks or dense matrix operations where linear algebra is the bottleneck.

*   **Configuration:**
    ```bash
    cmake -S . -B build -DRCLIB_USE_OPENMP=ON -DRCLIB_ENABLE_EIGEN_PARALLELIZATION=ON
    ```

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

2.  **Visualize Results:**
    ```bash
    uv run python benchmarks/plot_parallel_comparison.py
    ```

## Architecture & API Reference

### Key Architectural Principles

1.  **Modularity:** The **Reservoir** and **Readout** components are implemented as separate, swappable modules.
2.  **Performance:** C++ implementations prioritize computational efficiency and memory management, especially for large, sparse matrices (`Eigen::SparseMatrix`).
3.  **Scalability:** Supports large reservoirs (40,000+ neurons), deep ESNs (serial stacking), and parallel ESNs.
4.  **Configurability:** Key parameters (spectral radius, sparsity, leak rate, regularization, bias, etc.) are configurable via C++ and Python APIs.

### C++ API Design

**Reservoir Interface (`Reservoir.h`)**
*   `virtual Eigen::MatrixXd advance(const Eigen::MatrixXd& input) = 0;`
*   `virtual void resetState() = 0;`
*   `virtual const Eigen::MatrixXd& getState() const = 0;`

**Readout Interface (`Readout.h`)**
*   `virtual void fit(const Eigen::MatrixXd& states, const Eigen::MatrixXd& targets) = 0;`
*   `virtual void partialFit(const Eigen::MatrixXd& state, const Eigen::MatrixXd& target) = 0;`
*   `virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& states) = 0;`

**Model Class (`Model.h`)**
*   Manages collections of reservoirs and readouts.
*   Supports `addReservoir` (serial/parallel) and `setReadout`.

### Python API

The Python interface (`rclib`) mirrors the C++ structure using `pybind11` and provides a `scikit-learn` compatible API.

**Example Usage:**

```python
from rclib import reservoirs, readouts
from rclib.model import ESN

# Configure
res1 = reservoirs.RandomSparse(n_neurons=1000, spectral_radius=0.9, include_bias=True)
readout = readouts.Ridge(alpha=1e-8, include_bias=True)

# Model
model = ESN(connection_type='serial')
model.add_reservoir(res1)
model.set_readout(readout)

# Train & Predict
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
