#!/usr/bin/env bash
# Set up the local development environment for rclib's Python bindings.
#
# The project uses scikit-build-core's rebuild-on-import (editable.rebuild)
# together with uv's no-build-isolation so that editing C++ and re-running
# `uv run python ...` recompiles automatically. That requires the build backend
# to be installed *before* the project itself is built, which uv cannot yet do
# in a single pass (https://github.com/astral-sh/uv/issues/13998). This script
# wraps the required two-step `uv sync` into one command.
#
# Any extra arguments are forwarded to the second `uv sync`, e.g.:
#   ./scripts/setup-dev.sh --group examples
set -euo pipefail

# Run from the repository root regardless of the caller's working directory.
cd "$(dirname "$0")/.."

echo "[1/2] Installing build backend (scikit-build-core, pybind11, cmake, ninja)..."
uv sync --no-install-project --only-group build

echo "[2/2] Building rclib and installing the development environment..."
uv sync "$@"

echo
echo "Done. The C++ extension now rebuilds automatically on the next import."
echo "Try: uv run python examples/python/quick_start.py"
