# Installation

## Prerequisites

*   **Operating System**: Linux (the only platform tested and supported).
*   **C++ Compiler**: Must support C++17 (e.g., GCC 9+, Clang 10+).
*   **CMake**: Version 3.15 or higher.
*   **Python**: Version 3.11 or higher.
*   **OpenMP**: Required for parallelization.
    *   Ubuntu/Debian: `sudo apt install libomp-dev`

## Installing with `uv` (Recommended)

The recommended way to install and manage `rclib` for development is using `uv`.

```bash
# Clone the repository
git clone --recursive https://github.com/hrshtst/rclib.git
cd rclib

# Install dependencies and the project (wraps the required two-step `uv sync`)
./scripts/setup-dev.sh
```

## Installing via `pip`

You can also install it using standard `pip`:

```bash
pip install .
```

*Note: The `--recursive` flag in git clone is crucial because `rclib` uses submodules for its C++ dependencies (Eigen, Catch2, pybind11).*
