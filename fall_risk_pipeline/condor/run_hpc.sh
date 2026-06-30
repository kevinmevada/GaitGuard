#!/bin/bash
# Run hpc.py commands on an OSPool worker (or locally on ap40).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ospool_paths.sh
source "${SCRIPT_DIR}/ospool_paths.sh"

STAGING="${STAGING_LOCAL}"
PROJECT="${GAITGUARD_PROJECT:-${HOME}/projects/GaitGuard/fall_risk_pipeline}"

export STAGING OSDF_BASE OSDF_GAITGUARD STAGING_LOCAL
export PYTHONHASHSEED=42

on_worker() {
  [[ -n "${_CONDOR_SCRATCH_DIR:-}" ]]
}

if on_worker; then
  export TMPDIR="${_CONDOR_SCRATCH_DIR}/tmp"
  mkdir -p "${TMPDIR}"
else
  export TMPDIR="${STAGING}/tmp"
  export PIP_CACHE_DIR="${STAGING}/pip-cache"
  export XDG_CACHE_HOME="${STAGING}/cache"
fi

setup_python() {
  if on_worker; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/bootstrap_worker_env.sh"
  else
    mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
    if [[ -x "${STAGING}/miniforge3/envs/gaitguard/bin/python" ]]; then
      # shellcheck source=/dev/null
      source "${STAGING}/miniforge3/bin/activate" "${STAGING}/miniforge3/envs/gaitguard"
    else
      # shellcheck source=/dev/null
      source "${STAGING}/miniforge3/bin/activate"
      conda activate gaitguard
    fi
    cd "${PROJECT}"
  fi
}

run_shard() {
  local stage="$1"
  shift
  local manifest=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--manifest" ]]; then
      manifest="$2"
      shift 2
    else
      shift
    fi
  done
  [[ -n "${manifest}" ]] || { echo "missing --manifest" >&2; exit 1; }

  if on_worker; then
    python condor/stage_shard_inputs.py "${stage}" --manifest "${manifest}"
  fi
  python hpc.py shard "${stage}" --manifest "${manifest}"
  if on_worker; then
    python condor/package_shard_outputs.py "${stage}" --manifest "${manifest}"
  fi
}

run_merge() {
  local stage="$1"
  if on_worker; then
    python condor/fetch_shards_for_merge.py "${stage}"
    mkdir -p data/processed data/features
  fi
  python hpc.py merge "${stage}"
  if on_worker; then
    python condor/publish_merge_outputs.py "${stage}"
  fi
}

setup_python
if on_worker; then
  echo "=== worker: $(hostname) scratch=${_CONDOR_SCRATCH_DIR} ==="
else
  echo "=== ap40 local: $(hostname) ==="
  cd "${PROJECT}"
fi

echo "=== hpc: $* ==="

case "${1:-}" in
  shard)
    run_shard "$2" "${@:3}"
    ;;
  merge|reduce)
    run_merge "$2"
    ;;
  *)
    python hpc.py "$@"
    ;;
esac

echo "=== done ==="
