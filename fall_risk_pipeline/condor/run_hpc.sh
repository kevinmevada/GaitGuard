#!/bin/bash
# Run hpc.py commands on an OSPool worker.
# Usage: run_hpc.sh manifests ingest | shard ingest --manifest PATH | merge ingest
set -euo pipefail

STAGING="/ospool/ap40/data/kevin.mevada"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"

export STAGING TMPDIR="${STAGING}/tmp" PIP_CACHE_DIR="${STAGING}/pip-cache"
export XDG_CACHE_HOME="${STAGING}/cache" PYTHONHASHSEED=42

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
source "${STAGING}/miniforge3/bin/activate"
conda activate gaitguard

cd "${PROJECT}"
echo "=== host: $(hostname) === hpc: $* ==="
python hpc.py "$@"
echo "=== done ==="
