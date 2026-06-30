#!/bin/bash
# Install miniforge + gaitguard env on OSDF staging for OSPool worker transfer.
# Run once on ap40 login node (takes ~10–20 min).
set -euo pipefail

STAGING="${STAGING:-/ospool/ap40/data/kevin.mevada}"
MAMBA_ROOT="${STAGING}/miniforge3"
ENV_NAME="${ENV_NAME:-gaitguard}"
ENV_PATH="${MAMBA_ROOT}/envs/${ENV_NAME}"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"
PY="${ENV_PATH}/bin/python"

export TMPDIR="${STAGING}/tmp"
export PIP_CACHE_DIR="${STAGING}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

if [[ ! -x "${MAMBA_ROOT}/bin/conda" ]]; then
  echo "Installing Miniforge to ${MAMBA_ROOT} ..."
  INSTALLER="$(mktemp /tmp/Miniforge3.XXXXXX.sh)"
  curl -fsSL -o "${INSTALLER}" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${INSTALLER}" -b -p "${MAMBA_ROOT}"
  rm -f "${INSTALLER}"
fi

# Avoid picking up a different conda/env from the login shell (e.g. ~/miniforge3).
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER || true
export CONDA_ENVS_PATH="${MAMBA_ROOT}/envs"
export CONDA_PKGS_DIRS="${STAGING}/conda/pkgs"
mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}"

# shellcheck source=/dev/null
source "${MAMBA_ROOT}/bin/activate" base

env_ok() {
  [[ -x "${PY}" ]] && "${PY}" -c "import pandas, numpy, pyarrow" 2>/dev/null
}

if [[ -d "${ENV_PATH}" ]] && ! env_ok; then
  echo "WARN: removing broken env at ${ENV_PATH}"
  conda env remove -y -p "${ENV_PATH}" 2>/dev/null || rm -rf "${ENV_PATH}"
fi

if ! env_ok; then
  echo "Creating env at ${ENV_PATH} (python 3.11) ..."
  conda create -y -p "${ENV_PATH}" python=3.11 pip
  # shellcheck source=/dev/null
  source "${MAMBA_ROOT}/bin/activate" "${ENV_PATH}"
  which python
  python -m pip install -r "${PROJECT}/requirements-lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: expected ${PY} after install. Diagnostics:" >&2
  ls -la "${ENV_PATH}/bin/" 2>&1 || ls -la "${ENV_PATH}/" 2>&1 || true
  conda env list >&2
  exit 1
fi

"${PY}" -c "import pandas, numpy, pyarrow; print('OK:', pandas.__version__)"

echo ""
echo "Worker transfer path (used by condor/*.sub):"
echo "  ${PY}"
echo "  osdf://${STAGING#/}/miniforge3/envs/${ENV_NAME}?recursive"
echo ""
echo "Activate locally:"
echo "  source ${MAMBA_ROOT}/bin/activate ${ENV_PATH}"
