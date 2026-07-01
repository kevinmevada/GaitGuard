#!/bin/bash
# Create a lightweight venv on OSPool workers (avoids transferring 2.5 GB conda env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${HPC_WORKER_STAGE:-ingest}"
case "${STAGE}" in
  features)
    REQ="${SCRIPT_DIR}/requirements-hpc-cpu.txt"
    MIN_PY_MAJOR=3
    MIN_PY_MINOR=10
    MAX_PY_MAJOR=3
    MAX_PY_MINOR=12
    VENV_TAG="features"
    PY_CANDIDATES=(python3.10 python3.11 python3.12)
    ;;
  *)
    REQ="${SCRIPT_DIR}/requirements-hpc-ingest.txt"
    MIN_PY_MAJOR=3
    MIN_PY_MINOR=9
    MAX_PY_MAJOR=3
    MAX_PY_MINOR=12
    VENV_TAG="ingest"
    # Explicit versions only — bare python3 may be 3.13+ without pyarrow wheels.
    PY_CANDIDATES=(python3.9 python3.10 python3.11 python3.12)
    ;;
esac

VENV_DIR="${WORKER_VENV:-${PWD}/.worker-venv-${VENV_TAG}}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  exit 0
fi

if [[ ! -f "${REQ}" ]]; then
  echo "ERROR: missing ${REQ}" >&2
  exit 1
fi

_py_version_ok() {
  "$1" -c "import sys
vi = sys.version_info
lo = (${MIN_PY_MAJOR}, ${MIN_PY_MINOR})
hi = (${MAX_PY_MAJOR}, ${MAX_PY_MINOR})
ok = lo <= (vi.major, vi.minor) <= hi
sys.exit(0 if ok else 1)" 2>/dev/null
}

_has_venv_module() {
  "$1" -c "import venv" 2>/dev/null && "$1" -m venv --help >/dev/null 2>&1
}

_pick_python() {
  local cand
  if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
    if command -v "${PYTHON_BOOTSTRAP}" >/dev/null 2>&1 && _py_version_ok "${PYTHON_BOOTSTRAP}"; then
      echo "${PYTHON_BOOTSTRAP}"
      return 0
    fi
  fi
  # Prefer interpreters that can run "python -m venv" without PEP 668 hacks.
  for cand in "${PY_CANDIDATES[@]}"; do
    if command -v "${cand}" >/dev/null 2>&1 && _py_version_ok "${cand}" && _has_venv_module "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  # Fallback: any version-ok interpreter (may need virtualenv + break-system-packages).
  for cand in "${PY_CANDIDATES[@]}"; do
    if command -v "${cand}" >/dev/null 2>&1 && _py_version_ok "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

_install_virtualenv() {
  local py="$1"
  if command -v virtualenv >/dev/null 2>&1; then
    return 0
  fi
  if "${py}" -m pip install --user virtualenv; then
    return 0
  fi
  echo "pip --user blocked (PEP 668); using --break-system-packages for virtualenv bootstrap only ..." >&2
  "${py}" -m pip install --user --break-system-packages virtualenv
}

_create_venv() {
  local py="$1"
  local venv_dir="$2"
  if _has_venv_module "${py}" && "${py}" -m venv "${venv_dir}" 2>/dev/null; then
    return 0
  fi
  echo "venv module missing for ${py}; trying virtualenv ..." >&2
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv -p "${py}" "${venv_dir}"
    return $?
  fi
  _install_virtualenv "${py}"
  "${py}" -m virtualenv -p "${py}" "${venv_dir}"
}

PY="$(_pick_python || true)"
if [[ -z "${PY}" ]]; then
  echo "ERROR: no Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}-${MAX_PY_MAJOR}.${MAX_PY_MINOR} with venv on worker (stage=${STAGE})." >&2
  for cand in "${PY_CANDIDATES[@]}" python3; do
    command -v "${cand}" >/dev/null 2>&1 && "${cand}" --version >&2 || true
  done
  exit 1
fi

echo "Bootstrapping worker venv [${STAGE}] with ${PY} ($(${PY} --version)) ..."
_create_venv "${PY}" "${VENV_DIR}"

"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
if ! "${VENV_DIR}/bin/pip" install --no-cache-dir -r "${REQ}"; then
  echo "ERROR: pip install failed for ${REQ}" >&2
  "${VENV_DIR}/bin/pip" --version >&2 || true
  exit 1
fi

export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -c "import pandas, numpy, pyarrow, scipy, yaml; print('worker venv OK [${STAGE}]')"
