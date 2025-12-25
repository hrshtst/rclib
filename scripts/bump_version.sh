#!/bin/bash
# Script to bump the version in pyproject.toml

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {major|minor|patch}"
    exit 1
fi

PART=$1
FILE="pyproject.toml"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found."
    exit 1
fi

# Extract the current version string (e.g., 0.0.1)
CURRENT_VERSION=$(grep '^version = ' "$FILE" | sed -E 's/version = "(.*)"/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: Could not find version string in $FILE"
    exit 1
fi

# Split version into major, minor, patch
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

case "$PART" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Error: Invalid part '$PART'. Use major, minor, or patch."
        exit 1
        ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"

# Update the file
# Note: Using a temporary file for cross-platform sed compatibility
sed -E "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$FILE" > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"

# Sync uv.lock
uv lock

echo "Successfully bumped version from $CURRENT_VERSION to $NEW_VERSION"

# Automate git commands
git add "$FILE" uv.lock
git commit -m "chore: bump version to $NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

echo ""
echo "Done! Version bumped, committed, and tagged as v$NEW_VERSION."
echo "To push the release, run:"
echo "  git push origin main --atomic --follow-tags"
