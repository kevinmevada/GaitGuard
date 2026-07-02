#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT}"

TRIALS="${TRIALS:-5}"
ROWS="${ROWS:-150}"
pick_worker_python() {
  local c
  for c in \
    "${PYTHON_BOOTSTRAP:-}" \
    "/c/Users/mevad/AppData/Local/Programs/Python/Python312/python.exe" \
    "/c/Users/mevad/AppData/Local/Programs/Python/Python311/python.exe" \
    "/c/Users/mevad/AppData/Local/Programs/Python/Python310/python.exe" \
    python3.12 python3.11 python3.10 python3 python; do
    [[ -n "${c}" ]] || continue
    if command -v "${c}" >/dev/null 2>&1; then
      "${c}" - <<'PYV' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)
PYV
      if [[ $? -eq 0 ]]; then
        echo "${c}"
        return 0
      fi
    fi
  done
  return 1
}
WORKER_PY="$(pick_worker_python || true)"
if [[ -z "${WORKER_PY}" ]]; then
  echo "ERROR: need Python 3.10-3.12 for local worker simulation" >&2
  exit 1
fi
PY="${PYTHON_DRIVER:-python}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  PY="${WORKER_PY}"
fi
export PYTHON_BOOTSTRAP="${WORKER_PY}"
STAGING="${PROJECT}/.local_staging/gaitguard"
LOG_DIR="${PROJECT}/logs/local_hpc"
mkdir -p "${LOG_DIR}" "${PROJECT}/data/hpc/shards/ingest" \
  "${PROJECT}/data/hpc/shards/preprocess" "${PROJECT}/data/hpc/shards/features"

export GAITGUARD_LOCAL_STAGING="${STAGING}"
export GAITGUARD_PROJECT="${PROJECT}"
export PYTHONHASHSEED=42
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export GAITGUARD_NO_PROGRESS=1
export PYTHONPATH="${PROJECT}/condor${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "merge pipeline config"
CONFIG="$("${PY}" scripts/merge_pipeline_config.py)"
export GAITGUARD_CONFIG="${CONFIG}"

log "seed synthetic raw data (${TRIALS} trials)"
"${PY}" scripts/seed_local_raw_data.py --trials "${TRIALS}" --rows "${ROWS}"

log "discover + validate_raw"
"${PY}" main.py --config "${CONFIG}" --stage discover
"${PY}" main.py --config "${CONFIG}" --stage validate_raw

log "hpc init + manifests"
"${PY}" hpc.py --config "${CONFIG}" init
rm -rf condor/manifests/local
"${PY}" hpc.py --config "${CONFIG}" manifests ingest
"${PY}" hpc.py --config "${CONFIG}" manifests preprocess
"${PY}" hpc.py --config "${CONFIG}" manifests features

run_shards() {
  local stage="$1"
  local pattern="$2"
  local manifest
  for manifest in condor/manifests/local/${pattern}; do
    [[ -f "${manifest}" ]] || continue
    log "worker shard ${stage}: ${manifest}"
    bash scripts/run_local_worker.sh shard "${stage}" --manifest "${manifest}" \
      2>&1 | tee "${LOG_DIR}/${stage}_$(basename "${manifest}" .json).log"
  done
}

log "ingest shards"
run_shards ingest "ingest_chunk_*.json"

log "merge ingest"
bash scripts/run_local_worker.sh merge ingest 2>&1 | tee "${LOG_DIR}/merge_ingest.log"
"${PY}" condor/extract_merge_bundle.py ingest

log "preprocess shards"
run_shards preprocess "preprocess_chunk_*.json"

log "merge preprocess"
bash scripts/run_local_worker.sh merge preprocess 2>&1 | tee "${LOG_DIR}/merge_preprocess.log"
"${PY}" condor/extract_merge_bundle.py preprocess

log "features shards"
run_shards features "features_chunk_*.json"

log "merge features"
bash scripts/run_local_worker.sh merge features 2>&1 | tee "${LOG_DIR}/merge_features.log"
"${PY}" condor/extract_merge_bundle.py features

log "verify outputs"
"${PY}" scripts/verify_local_hpc_outputs.py --config "${CONFIG}"

log "=== LOCAL HPC PIPELINE COMPLETE ==="
