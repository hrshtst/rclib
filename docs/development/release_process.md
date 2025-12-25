# Release Process

This document outlines the steps required to create a new release of `rclib`, including publishing to PyPI and updating the documentation.

## Prerequisites

1.  **Permissions:** You must have write access to the GitHub repository and be configured as a **Trusted Publisher** on PyPI for this repository.
2.  **Environment:** Ensure all tests pass locally using `uv run nox`.

## Step-by-Step Guide

### 1. Update Version Number
Use the provided script to increment the version in `pyproject.toml`, sync the lockfile, and create a git commit and tag automatically.

```bash
# Choose one based on the change type
./scripts/bump_version.sh patch
# or
./scripts/bump_version.sh minor
# or
./scripts/bump_version.sh major
```

### 2. Push to GitHub
After the script completes, push the new commit and tag to GitHub.

```bash
git push origin main --atomic --follow-tags
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
*   It builds the source distribution (`.tar.gz`).
*   It builds **manylinux-compliant binary wheels** for multiple Python versions (3.11, 3.12, 3.13) using `cibuildwheel`.
*   It securely uploads all artifacts to PyPI using OpenID Connect (OIDC).

### 5. Documentation Deployment
The documentation is automatically deployed to GitHub Pages whenever changes are merged into the `main` branch. If your release involved merging into `main`, your documentation at [https://hrshtst.github.io/rclib/](https://hrshtst.github.io/rclib/) will be updated.

## Versioning Policy
`rclib` follows [Semantic Versioning (SemVer)](https://semver.org/):
*   **MAJOR** version for incompatible API changes.
*   **MINOR** version for add functionality in a backwards compatible manner.
*   **PATCH** version for backwards compatible bug fixes.
