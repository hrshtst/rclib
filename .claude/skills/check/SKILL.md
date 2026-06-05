---
name: check
description: Run the full local verification suite for rclib (lint, type-check, Python + C++ tests) before committing. Use when asked to verify changes, run checks, or confirm the build is green.
---

Run rclib's verification suite, the same checks CI enforces. Run each step and report pass/fail with the relevant output; do not stop at the first failure unless asked — collect all results so the user sees the full picture.

1. **Lint & format** — `uv run nox -s lint`
   (ruff check + ruff format --check + shellcheck + cmake-format)
2. **Type check** — `uv run nox -s type_check`
   (basedpyright; this also builds the `_rclib` extension into the nox venv)
3. **Python tests** — `uv run nox -s tests`
   (builds the extension, runs pytest with coverage; slow tests are excluded by default)
4. **C++ tests** — `uv run nox -s tests_cpp`
   (configures with `-DBUILD_TESTING=ON`, builds, runs ctest)

Notes:
- If `$ARGUMENTS` names a subset (e.g. `python`, `cpp`, `lint`), run only those steps.
- To include slow tests, run `uv run pytest -m "slow or not slow"` after the suite.
- Fixable lint issues: `uv run ruff check --fix` and `uv run ruff format`.
- A first nox run builds fresh virtualenvs and may take a while; reuse is enabled for subsequent runs.
