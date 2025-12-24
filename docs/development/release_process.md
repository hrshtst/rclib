# Release Process

This document outlines the steps required to create a new release of `rclib`, including publishing to PyPI and updating the documentation.

## Prerequisites

1.  **Permissions:** You must have write access to the GitHub repository and be configured as a **Trusted Publisher** on PyPI for this repository.
2.  **Environment:** Ensure all tests pass locally using `uv run nox`.

## Step-by-Step Guide

### 1. Update Version Number
Use the provided script to increment the version in `pyproject.toml`.

```bash
# Choose one based on the change type
./scripts/bump_version.sh patch
# or
./scripts/bump_version.sh minor
# or
./scripts/bump_version.sh major
```

Follow the printed instructions to commit the change:
```bash
git add pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
```

### 2. Tag and Push
When you are ready to release, create a new semantic version tag and push it to GitHub.

```bash
# Example for version 0.1.0
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 3. Review the Release Draft
Pushing a tag starting with `v*` automatically triggers the **Create Release Draft** workflow.

1.  Go to the **Releases** section of the GitHub repository.
2.  Find the new draft release.
3.  Review the automatically generated release notes.
4.  Click **Edit** to add any manual highlights or breaking change notices.
5.  Click **Publish release**.

### 4. Automated Publishing
Once the release is published on GitHub, the **Publish to PyPI** workflow triggers automatically:

*   It checks out the code (including submodules).
*   It builds the source distribution and the C++ binary wheels.
*   It securely uploads the artifacts to PyPI using OpenID Connect (OIDC).

### 5. Documentation Deployment
The documentation is automatically deployed to GitHub Pages whenever changes are merged into the `main` branch. If your release involved merging into `main`, your documentation at [https://hrshtst.github.io/rclib/](https://hrshtst.github.io/rclib/) will be updated.

## Versioning Policy
`rclib` follows [Semantic Versioning (SemVer)](https://semver.org/):
*   **MAJOR** version for incompatible API changes.
*   **MINOR** version for add functionality in a backwards compatible manner.
*   **PATCH** version for backwards compatible bug fixes.
