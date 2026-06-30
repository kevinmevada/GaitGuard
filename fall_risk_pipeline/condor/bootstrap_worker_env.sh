#!/bin/bash
# Create a lightweight venv on OSPool workers (avoids transferring 2.5 GB conda env).
set -euo pipefail

VENV_DIR="${WORKER_VENV:-${PWD}/.worker-venv}"
REQ="${PWD}/condor/requirements-hpc-cpu.txt"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  exit 0
fi

if [[ ! -f "${REQ}" ]]; then
  echo "ERROR: missing ${REQ}" >&2
  exit 1
fi

PY="${PYTHON_BOOTSTRAP:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  PY=python
fi

echo "Bootstrapping worker venv with ${PY} ..."
"${PY}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel
"${VENV_DIR}/bin/pip" install --no-cache-dir -r "${REQ}"
export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -c "import pandas, numpy, pyarrow, scipy, yaml; print('worker venv OK')"
