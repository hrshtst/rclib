"""Nox configuration for rclib."""

from __future__ import annotations

import os

import nox

# Define the supported Python versions
PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]

nox.options.sessions = ["lint", "type_check", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True


@nox.session(venv_backend="none")
def dev(session: nox.Session) -> None:
    """Provision the local dev environment with fast incremental C++ rebuilds.

    Runs the two-step ``uv sync`` required by the no-build-isolation +
    rebuild-on-import setup (https://github.com/astral-sh/uv/issues/13998), so a
    single ``uv run nox -s dev`` prepares the project ``.venv``. This operates on
    the project environment directly (no nox-managed venv). Extra arguments are
    forwarded to the second sync, e.g. ``uv run nox -s dev -- --group examples``.
    """
    session.run("uv", "sync", "--no-install-project", "--only-group", "build", external=True)
    session.run("uv", "sync", *session.posargs, external=True)


@nox.session(python="3.12", reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install("ruff", "shellcheck-py", "cmakelang", "PyYAML")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("shellcheck", "scripts/bump_version.sh", "docs/development/reports/generate_pdf.sh")
    session.run("cmake-format", "--check", "CMakeLists.txt", "cpp_core/CMakeLists.txt", "python/CMakeLists.txt")


@nox.session(venv_backend="none")
def type_check(session: nox.Session) -> None:
    """Run type checking against the project environment.

    basedpyright resolves imports against the project's ./.venv (see
    [tool.basedpyright] venvPath/venv) and also checks examples/ and benchmarks/,
    which import the example/benchmark dependency groups. Run via `uv run
    --all-groups` so ./.venv is synced with every group (the default sync is light
    and omits those stacks). This deliberately uses no nox-managed venv: otherwise
    the sync would target the session venv, not the ./.venv basedpyright reads.
    """
    session.run("uv", "run", "--all-groups", "basedpyright", external=True)


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run the Python test suite."""
    session.install("scikit-build-core", "pybind11")
    session.install("--no-build-isolation", ".")
    session.install("pytest", "pytest-cov", "pytest-randomly", "pytest-xdist")
    session.run(
        "pytest",
        "-n",
        "auto",
        "--cov=rclib",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",
        *session.posargs,
    )


@nox.session(python="3.12", reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the documentation."""
    session.install("scikit-build-core", "pybind11")
    session.install("--no-build-isolation", ".")
    session.install("mkdocs", "mkdocs-material", "mkdocstrings[python]", "pymdown-extensions")
    session.run("mkdocs", "build")


@nox.session(reuse_venv=True)
def tests_cpp(session: nox.Session) -> None:
    """Build and run C++ tests."""
    configure = ["cmake", "-S", ".", "-B", "build_nox", "-DBUILD_TESTING=ON", "-DRCLIB_USE_OPENMP=ON"]
    # CI sets RCLIB_WERROR=ON to fail the build on warnings in our own sources.
    if os.environ.get("RCLIB_WERROR"):
        configure.append("-DRCLIB_WERROR=ON")
    session.run(*configure)
    session.run("cmake", "--build", "build_nox", "--config", "Release", "-j")
    session.run("ctest", "--test-dir", "build_nox", "--output-on-failure")


@nox.session(reuse_venv=True)
def tests_cpp_sanitizers(session: nox.Session) -> None:
    """Build and run C++ tests under AddressSanitizer + UndefinedBehaviorSanitizer.

    OpenMP is disabled to avoid sanitizer false positives from the OpenMP runtime,
    isolating real memory/UB issues in the core numerical code.
    """
    san_flags = "-fsanitize=address,undefined -fno-omit-frame-pointer -g"
    session.run(
        "cmake",
        "-S",
        ".",
        "-B",
        "build_asan",
        "-DBUILD_TESTING=ON",
        "-DRCLIB_USE_OPENMP=OFF",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_CXX_FLAGS={san_flags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={san_flags}",
    )
    session.run("cmake", "--build", "build_asan", "-j")
    session.run(
        "ctest",
        "--test-dir",
        "build_asan",
        "--output-on-failure",
        env={
            "ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1",
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        },
    )
