#!/bin/bash
# Create a lightweight venv on OSPool workers (avoids transferring 2.5 GB conda env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${WORKER_VENV:-${PWD}/.worker-venv}"
REQ="${SCRIPT_DIR}/requirements-hpc-cpu.txt"
MIN_PY_MAJOR=3
MIN_PY_MINOR=10

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  exit 0
fi

if [[ ! -f "${REQ}" ]]; then
  echo "ERROR: missing ${REQ}" >&2
  exit 1
fi

_pick_python() {
  local cand
  if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
    if command -v "${PYTHON_BOOTSTRAP}" >/dev/null 2>&1 \
      && "${PYTHON_BOOTSTRAP}" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
      echo "${PYTHON_BOOTSTRAP}"
      return 0
    fi
  fi
  for cand in python3.12 python3.11 python3.10; do
    if command -v "${cand}" >/dev/null 2>&1 \
      && "${cand}" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
      echo "${cand}"
      return 0
    fi
  done
  # Last resort: default python3 if it meets the minimum (skip 3.9-only workers).
  if command -v python3 >/dev/null 2>&1 \
    && python3 -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
    echo "python3"
    return 0
  fi
  return 1
}

PY="$(_pick_python || true)"
if [[ -z "${PY}" ]]; then
  echo "ERROR: no Python >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR} on worker." >&2
  command -v python3 >/dev/null 2>&1 && python3 --version >&2 || true
  exit 1
fi

echo "Bootstrapping worker venv with ${PY} ($(${PY} --version)) ..."
if ! "${PY}" -m venv "${VENV_DIR}" 2>/dev/null; then
  echo "venv module missing; trying virtualenv ..."
  "${PY}" -m pip install --user virtualenv
  "${PY}" -m virtualenv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install --upgrade pip wheel
if ! "${VENV_DIR}/bin/pip" install --no-cache-dir -r "${REQ}"; then
  echo "ERROR: pip install failed for ${REQ}" >&2
  "${VENV_DIR}/bin/pip" --version >&2 || true
  exit 1
fi

export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -c "import pandas, numpy, pyarrow, scipy, yaml; print('worker venv OK')"
