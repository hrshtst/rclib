# AGENTS.md

Guidance for AI coding assistants working in this repository. Keep this file concise; deep design notes live in `docs/`.

## Project

**rclib** is a reservoir-computing framework: a C++17 core (Eigen) exposed to Python via `pybind11`. C++ lives in `cpp_core/`, the Python package in `python/rclib/` (the extension module is `_rclib`, built from `python/rclib/_rclib.cpp`). Architecture is modular and swappable: **Reservoir** (RandomSparse, NVAR) + **Readout** (Ridge, RLS, LMS) + **Model** (ESN, serial/parallel stacking). The library auto-selects solvers and (for `N > 1000`) switches to parallel reservoir updates.

## Setup

- Clone with submodules — third-party deps (Eigen, Catch2, pybind11) live in `cpp_core/third_party/`: `git clone --recursive` (or `git submodule update --init --recursive`).
- Python ≥ 3.11. `uv` manages the environment. OpenMP (`libomp-dev`) is needed for multithreading.

## Build

Python bindings (editable, with auto-rebuild on C++ changes) — run both steps, in order:

```bash
uv sync --no-install-project --only-group build   # build deps first
uv sync                                            # then the project
```

After this, editing any `cpp_core/` source triggers a rebuild on the next `import rclib`. C++ core / examples standalone:

```bash
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build --config Release -j $(nproc)
```

## Test

- Python: `uv run pytest` — slow tests are excluded by default (`-m "not slow"` is configured). Run everything with `uv run pytest -m "slow or not slow"`.
- C++ (Catch2): build with `-DBUILD_TESTING=ON`, then `ctest --test-dir build --output-on-failure`. Slow C++ tests are tagged `[.slow]` and run by passing the tag to a test binary, e.g. `./build/tests/cpp/test_readout "[.slow]"`.
- `nox` wraps the full matrix: `uv run nox` runs `lint`, `type_check`, `tests`. C++ tests: `uv run nox -s tests_cpp`.

## Lint, format & type-check

- `uv run nox -s lint` — ruff check + ruff format check + shellcheck + cmake-format.
- `uv run nox -s type_check` — basedpyright.
- `uv run pre-commit install` once, then hooks run on every commit (`pre-commit run -a` to run manually).

## Code style

- **Python:** ruff, line length **120**, `select = ["ALL"]` (see `pyproject.toml` for ignores). NumPy-style docstrings, double quotes. Every module must start with `from __future__ import annotations` (ruff `required-imports` enforces this).
- **C++:** C++17, clang-format **LLVM** style (`.clang-format`). Use Eigen for linear algebra. Do not touch `cpp_core/third_party/`.
- CMake files are formatted with cmake-format (`.cmake-format.yaml`).

## Conventions

- **Commits:** Conventional Commits — `feat:`, `fix:`, `docs:`, `test:`, `perf:`, `chore:`, `style:`, `benchmark:`.
- **CI** runs on `main` and `develop` (Python tests across 3.11–3.13, C++ tests, lint + type-check). Keep these green.
- Build flags (`RCLIB_USE_OPENMP`, `RCLIB_ENABLE_EIGEN_PARALLELIZATION`, `RCLIB_ADAPTIVE_PARALLELIZATION`) default to `ON`; see `README.md` for tuning.
