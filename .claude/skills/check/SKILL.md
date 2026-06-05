---
name: check
description: Run rclib's local verification suite — ruff lint/format, basedpyright type-check, pytest, and the C++ ctest build — and report whether the branch is green. Use this whenever the user wants to confirm changes are good before committing, pushing, or opening a PR — e.g. "run the checks", "verify my change", "make sure nothing's broken", "is the build/branch green", or "did I break anything?" after editing C++ core or Python code. Not for writing or speeding up individual tests, debugging a single type or build error, formatting one file, reviewing a diff for bugs, authoring CI config, running benchmarks, or running pre-commit directly.
---

Run rclib's verification suite — the same checks CI enforces (lint, type-check, Python tests, C++ tests). The goal is a clear, at-a-glance answer to "is this branch green?" before committing.

## 1. Scope to what changed

Compiling the C++ core and building the pybind11 extension is the slow part, so don't blindly run everything — look at what actually changed (`git status --short`) and run only the steps a change can affect. Running the full suite for a one-line docstring edit wastes minutes.

| Changed | Run |
| --- | --- |
| Python only (`python/`, `tests/python/`, `*.py`) | lint, type-check, Python tests |
| C++ only (`cpp_core/`, `tests/cpp/`) | lint, C++ tests (add type-check + Python tests if the Python-facing API changed) |
| Shell / CMake (`*.sh`, `CMakeLists.txt`, `*.cmake`) | lint (via nox — see step 4) |
| Docs / config only | lint |
| Unsure, or preparing a release/PR | everything, via nox (step 4) |

If the user passes a subset in `$ARGUMENTS` (`lint`, `types`, `python`/`py`, `cpp`, `all`), honor that instead of inferring from the diff.

## 2. Run the relevant steps (fast path)

Default to the dev environment: `.venv` already has the tools and the editable install auto-rebuilds `_rclib` on import, so these skip nox's per-session venv setup and are much faster on repeat runs. Run in order — lint is fastest and catches the most, so it gives the earliest signal. **Don't stop at the first failure** (unless asked): collect every step's result so the user can fix everything in one pass instead of round-tripping.

- **Lint & format:** `uv run ruff check .` then `uv run ruff format --check .`
  (Fast, ruff-only. If shell or CMake files changed, run `uv run nox -s lint` instead — it adds shellcheck + cmake-format, which the fast path skips.)
- **Type check:** `uv run basedpyright`
- **Python tests:** `uv run pytest -n auto`  (slow tests are excluded by default)
- **C++ tests:** reuse the persistent `build/` dir for incremental compiles:
  ```bash
  cmake -S . -B build -DBUILD_TESTING=ON
  cmake --build build --config Release -j
  ctest --test-dir build --output-on-failure
  ```

## 3. Report

End with a one-line-per-step summary so the verdict is scannable:

```
lint        ✓
type_check  ✓
python      ✗  2 failed — tests/python/test_rls.py::test_woodbury, ...
cpp         —  skipped (no C++ changes)
```

For a failure, surface the relevant error/assertion, not the whole log.

## 4. Full CI parity (when it matters)

The fast path uses your local environment; CI runs in clean, isolated venvs and can catch env-specific breakage the fast path misses. Before a release, when opening a PR, or to reproduce a CI failure, run the nox sessions instead — they mirror CI exactly:

`uv run nox` (lint + type_check + tests) and `uv run nox -s tests_cpp`.

## Notes

- **Auto-fix lint:** `uv run ruff check --fix` and `uv run ruff format`.
- **Include slow tests:** `uv run pytest -m "slow or not slow"` (Python), or pass `[.slow]` to a C++ binary, e.g. `./build/tests/cpp/test_readout "[.slow]"`.
- The first nox run builds fresh virtualenvs and is slow; reuse is enabled for later runs.
