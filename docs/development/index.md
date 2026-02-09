# Development Guide

This section is for contributors who want to modify or extend `rclib`.

## Project Goals

*   **Performance**: Core logic in C++17 using Eigen.
*   **Scalability**: Efficient handling of large sparse reservoirs.
*   **Modularity**: Clear separation between Reservoirs and Readouts.

## Setup

1.  **Install `uv`**: We use `uv` for dependency management.
2.  **Clone recursively**: `git clone --recursive ...`
3.  **Sync environment**: `uv sync`

## Workflow

### Code Quality

We use `pre-commit` to enforce standards.

```bash
uv run pre-commit install
```

Tools used:
*   `ruff` (Python linting/formatting)
*   `basedpyright` (Static type checking)
*   `clang-format` (C++ formatting)

### Running Tests

**C++ Unit Tests:**
```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build
```

**Python Integration Tests:**
```bash
# Ensure library is built
cmake -S . -B build
cmake --build build --config Release -j $(nproc) --target _rclib
# Run pytest
uv run pytest
```

## Documentation

*   [Release Process](release_process.md): How to publish a new version.
*   [Testing Roadmap](testing_roadmap.md): Plans for future test coverage.
*   [RLS Optimization Report](reports/RLS_Optimization_Report.md): Detailed report on RLS performance improvements.
