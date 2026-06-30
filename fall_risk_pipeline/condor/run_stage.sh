#!/bin/bash
# Run one GaitGuard pipeline stage on an OSPool worker.
# Usage: run_stage.sh <stage_name>   e.g.  run_stage.sh ingest
set -euo pipefail

STAGE="${1:?usage: run_stage.sh <stage_name>}"
STAGING="/ospool/ap40/data/kevin.mevada"
PROJECT="${GAITGUARD_PROJECT:-${HOME}/projects/GaitGuard/fall_risk_pipeline}"

export STAGING
export TMPDIR="${STAGING}/tmp"
export PIP_CACHE_DIR="${STAGING}/pip-cache"
export XDG_CACHE_HOME="${STAGING}/cache"
export HF_HOME="${STAGING}/cache/huggingface"
export PYTHONHASHSEED=42
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-${REQUEST_CPUS:-4}}"

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

source "${STAGING}/miniforge3/bin/activate"
conda activate gaitguard

cd "${PROJECT}"
echo "=== host: $(hostname) ==="
echo "=== date: $(date -Is) ==="
echo "=== stage: ${STAGE} ==="
echo "=== python: $(which python) ==="
echo "=== cwd: $(pwd) ==="

python main.py --stage "${STAGE}"

echo "=== done: ${STAGE} at $(date -Is) ==="
