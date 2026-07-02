#!/usr/bin/env bash
# Syntax-check all shell scripts under fall_risk_pipeline/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
fail=0
while IFS= read -r -d '' f; do
  if bash -n "${f}"; then
    echo "OK  ${f#${ROOT}/}"
  else
    echo "FAIL ${f#${ROOT}/}"
    fail=1
  fi
done < <(find . -name '*.sh' -not -path './.local_staging/*' -print0)
if command -v shellcheck >/dev/null 2>&1; then
  echo "=== shellcheck ==="
  shellcheck $(find . -name '*.sh' -not -path './.local_staging/*') || fail=1
else
  echo "shellcheck not installed (skipped)"
fi
exit "${fail}"
