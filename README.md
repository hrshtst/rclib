# rclib: Reservoir Computing Library

**rclib** is a high-performance, scalable, and general-purpose reservoir computing framework implemented in C++ with Python bindings. It is designed to handle both small-scale networks and very large-scale (40,000+ neurons) architectures, supporting deep (stacked) and parallel reservoir configurations.

## Project Goals

*   **Performance:** Core logic in C++17 using Eigen for linear algebra.
*   **Scalability:** Efficient handling of large sparse reservoirs and complex architectures.
*   **Flexibility:** Modular design separating Reservoirs and Readouts.
*   **Ease of Use:** Pythonic interface via `pybind11` and `scikit-learn` style API.
*   **Reproducibility:** Deterministic results via explicit seeding of random reservoirs.

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

### Using Nox (Recommended)
`nox` automates environment setup and execution for both Python and C++.

```bash
# Run all default sessions (Lint, Type Check, Python Tests)
uv run nox

# Run Python tests only
uv run nox -s tests

# Run C++ tests only
uv run nox -s tests_cpp
```

### Manual Execution (Step-by-Step)

If you prefer to run tests manually without `nox`:

#### C++ Tests
```bash
# 1. Configure and build
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j $(nproc)

# 2. Run tests
ctest --test-dir build --output-on-failure
```

#### Python Tests
```bash
# 1. Build and install the C++ extension in the current environment
cmake -S . -B build
cmake --build build --config Release -j $(nproc) --target _rclib

# 2. Run pytest
uv run pytest
```

## Documentation

The project documentation is built using `mkdocs` and the Material theme. It includes theoretical background, user guides, and API references.

### Using Nox (Recommended)
```bash
# Build the documentation
uv run nox -s docs
```

### Manual Execution
If you prefer to run `mkdocs` directly:

```bash
# 1. Install documentation dependencies
uv sync --group docs

# 2. Build the documentation
uv run mkdocs build

# 3. Serve the documentation locally with live-reloading
uv run mkdocs serve
```

The documentation is automatically deployed to [https://hrshtst.github.io/rclib/](https://hrshtst.github.io/rclib/) on every push to the `main` branch.

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
    uv run python benchmarks/plot_parallel_comparison.py
    ```
    This generates plots comparing execution time and MSE for different methods and configurations.

## Development

### Code Quality Tools

The project uses several tools to ensure code quality, all of which are integrated into `pre-commit` and `nox`:

*   **Ruff:** For Python linting and formatting.
*   **Basedpyright:** For static type checking.
*   **clang-format:** For C++ formatting (LLVM style).
*   **shellcheck:** For shell script linting.
*   **cmake-format / cmake-lint:** For CMake formatting and linting.
*   **pre-commit:** To enforce checks before committing.

### Automation with Nox

`nox` is used to automate various development tasks:

*   `uv run nox -s lint`: Run linters.
*   `uv run nox -s type_check`: Run type checkers.
*   `uv run nox -s tests`: Run Python tests.
*   `uv run nox -s tests_cpp`: Run C++ tests.
*   `uv run nox -s docs`: Build documentation.

### Setting up pre-commit

To ensure all code follows the project's style and quality standards, it is recommended to set up `pre-commit`:

```bash
uv run pre-commit install
```

## AI Assistance & Development Workflow

This project is developed with the assistance of an AI coding agent using the [Gemini CLI](https://github.com/google-gemini/as-code) tool. The AI is also used to generate commit messages and parts of the documentation, including API and theoretical reference sections.

**Workflow:**
1.  **Context & Theory (Human):** The maintainer, **[Hiroshi Atsuta](https://github.com/hrshtst)**, establishes the project roadmap in `GEMINI.md` and writes the theoretical background implemented as documentation in [docs/theory/](docs/theory/).
2.  **Implementation (AI):** The AI assistant uses these documents and the constraints defined in `GEMINI.md` to implement code scaffolding, core logic, tests, and documentation.
3.  **Review & Revision (Human):** The maintainer reviews, tests, and revises the generated code to ensure quality and correctness. This iterative cycle ensures high standards while leveraging AI efficiency.

**Responsibility:**
All responsibilities for the code hosted in this repository lie with the maintainer. The AI serves strictly as an implementation assistant; final architectural decisions and code quality are human-led.

**Feedback:**
If you identify problems, or find code that appears to be unoriginal or rights-protected, please notify the maintainer immediately by filing an issue.

**Contributor Policy:**
External contributors are welcome to use AI tools for assistance, provided they adhere to the same standard of review and responsibility. If you use AI to generate code for a Pull Request, please disclose it in the PR description and ensure you have thoroughly reviewed and tested the code.

## Acknowledgments

<img src="docs/assets/ipa_logo.png" height="60" alt="IPA Logo"> &nbsp; &nbsp; <img src="docs/assets/mitou_target_logo.png" height="60" alt="MITOU Target Logo">

This project is supported by the **[MITOU Target Program](https://www.ipa.go.jp/jinzai/mitou/koubo/programs/target.html)** (Reservoir Computing field) of the [Information-technology Promotion Agency, Japan (IPA)](https://www.ipa.go.jp/en/index.html). Details of the supported project can be found in the [official summary](https://www.ipa.go.jp/jinzai/mitou/target/2025_reservoir/gaiyou-ky-1.html) (Japanese).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
