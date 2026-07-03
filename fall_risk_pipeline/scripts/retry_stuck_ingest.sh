#!/bin/bash
# Re-submit incomplete ingest shards after bootstrap fix — keeps finished shard tarballs.
#
# Usage (ap40):
#   cd ~/projects/GaitGuard/fall_risk_pipeline
#   bash scripts/retry_stuck_ingest.sh
set -euo pipefail

OWNER="$(whoami)"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"
GG="/ospool/ap40/data/kevin.mevada/gaitguard"
DAG="${PROJECT}/condor/dags/gaitguard_sharded.dag"

cd "${HOME}/projects/GaitGuard"
echo "=== git pull ==="
git pull

cd "${PROJECT}"
# shellcheck source=/dev/null
source /ospool/ap40/data/kevin.mevada/miniforge3/bin/activate \
  /ospool/ap40/data/kevin.mevada/miniforge3/envs/gaitguard
export TMPDIR=/ospool/ap40/data/kevin.mevada/tmp

# Jobs transfer the prebuilt env from OSDF; without it every node holds on
# input transfer. Build once with: bash scripts/build_worker_env.sh
WORKER_ENV_TARBALL="$(grep -m1 -oP 'WORKER_ENV\s*=\s*\K\S+' condor/hpc_shard_ingest.sub || true)"
if [[ -n "${WORKER_ENV_TARBALL}" && ! -f "${GG}/worker-env/${WORKER_ENV_TARBALL}.tar.gz" ]]; then
  echo "ERROR: ${GG}/worker-env/${WORKER_ENV_TARBALL}.tar.gz not found."
  echo "Build it first:  bash scripts/build_worker_env.sh"
  exit 1
fi

mkdir -p "${GG}/hpc/shards/ingest"
done_shards="$(find "${GG}/hpc/shards/ingest" -maxdepth 1 -name '*.tar.gz' 2>/dev/null | wc -l | tr -d ' ')"
echo "=== ingest shards on OSDF: ${done_shards} / 68 ==="

echo "=== remove queued/running/held jobs (DAG rescue preserves completed nodes) ==="
condor_rm "${OWNER}" 2>/dev/null || true
sleep 2
if [[ "$(condor_q "${OWNER}" 2>/dev/null | awk 'NR>2 && NF {c++} END {print c+0}')" -gt 0 ]]; then
  condor_rm -forcex "${OWNER}" 2>/dev/null || true
  sleep 2
fi

echo "=== regenerate DAG (updated bootstrap in transfer_input_files) ==="
python hpc/submit/generate_dag.py --config configs/pipeline_config.yaml --skip-existing-ingest

echo "=== resubmit DAG (rescue file skips finished ing_* nodes) ==="
condor_submit_dag -f "${DAG}"

echo ""
echo "Monitor:"
echo "  ls ${GG}/hpc/shards/ingest/*.tar.gz | wc -l"
echo "  condor_q -dag ${OWNER}"
echo "  condor_tail <cluster>.0"
