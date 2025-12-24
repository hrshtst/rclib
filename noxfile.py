"""Nox configuration for rclib."""

from __future__ import annotations

import nox

# Define the supported Python versions
PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]

nox.options.sessions = ["lint", "type_check", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True


@nox.session(python="3.12", reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.12", reuse_venv=True)
def type_check(session: nox.Session) -> None:
    """Run type checking."""
    session.install("scikit-build-core", "pybind11")
    session.install("--no-build-isolation", ".")
    session.install("basedpyright", "matplotlib", "numpy", "pandas", "seaborn")
    session.run("basedpyright")


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
    session.run("cmake", "-S", ".", "-B", "build_nox", "-DBUILD_TESTING=ON", "-DRCLIB_USE_OPENMP=ON")
    session.run("cmake", "--build", "build_nox", "--config", "Release", "-j")
    session.run("ctest", "--test-dir", "build_nox", "--output-on-failure")
