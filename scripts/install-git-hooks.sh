#!/usr/bin/env bash
# Install repo git hooks (commit-msg strips Cursor attribution trailers).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK_SRC="$ROOT/scripts/git-hooks/commit-msg"
HOOK_DST="$ROOT/.git/hooks/commit-msg"
cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
PREP_SRC="$ROOT/scripts/git-hooks/prepare-commit-msg"
PREP_DST="$ROOT/.git/hooks/prepare-commit-msg"
cp "$PREP_SRC" "$PREP_DST"
chmod +x "$PREP_DST"
echo "Installed $HOOK_DST and $PREP_DST"
