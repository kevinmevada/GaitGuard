#!/bin/bash
# Install miniforge + gaitguard env on OSDF staging for OSPool worker transfer.
# Run once on ap40 login node (takes ~10–20 min).
set -euo pipefail

STAGING="${STAGING:-/ospool/ap40/data/kevin.mevada}"
MAMBA_ROOT="${STAGING}/miniforge3"
ENV_NAME="${ENV_NAME:-gaitguard}"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"

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

# shellcheck source=/dev/null
source "${MAMBA_ROOT}/bin/activate"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists at ${MAMBA_ROOT}/envs/${ENV_NAME}"
else
  echo "Creating ${ENV_NAME} (python 3.11) ..."
  conda create -y -n "${ENV_NAME}" python=3.11 pip
  conda activate "${ENV_NAME}"
  pip install -r "${PROJECT}/requirements-lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

PY="${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"
"${PY}" -c "import pandas, numpy, pyarrow; print('OK:', pandas.__version__)"

echo ""
echo "Worker transfer path (used by condor/*.sub):"
echo "  ${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"
echo "  osdf://${STAGING#/*}/miniforge3/envs/${ENV_NAME}?recursive"
echo ""
echo "Activate locally: source ${MAMBA_ROOT}/bin/activate && conda activate ${ENV_NAME}"
