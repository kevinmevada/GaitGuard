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

# HTCondor may flatten transfer_input_files; resolve script/manifest paths.
resolve_path() {
  local name="$1"
  if [[ -f "${name}" ]]; then
    echo "${name}"
    return 0
  fi
  if [[ -f "${SCRIPT_DIR}/${name}" ]]; then
    echo "${SCRIPT_DIR}/${name}"
    return 0
  fi
  if [[ -f "${PWD}/${name}" ]]; then
    echo "${PWD}/${name}"
    return 0
  fi
  local base
  base="$(basename "${name}")"
  if [[ -f "${SCRIPT_DIR}/${base}" ]]; then
    echo "${SCRIPT_DIR}/${base}"
    return 0
  fi
  if [[ -f "${PWD}/${base}" ]]; then
    echo "${PWD}/${base}"
    return 0
  fi
  echo "ERROR: cannot find ${name} (pwd=${PWD}, script_dir=${SCRIPT_DIR})" >&2
  return 1
}

condor_py() {
  local script
  script="$(resolve_path "$1")" || return 1
  shift
  python "${script}" "$@"
}

if on_worker; then
  cd "${_CONDOR_SCRATCH_DIR}"
  export TMPDIR="${_CONDOR_SCRATCH_DIR}/tmp"
  mkdir -p "${TMPDIR}" data/hpc/shards data/processed data/raw
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
  manifest="$(resolve_path "${manifest}")" || exit 1

  local rc=0
  local err_msg=""

  if on_worker; then
    condor_py stage_shard_inputs.py "${stage}" --manifest "${manifest}" || {
      rc=$?
      err_msg="stage_shard_inputs failed (rc=${rc})"
    }
  fi

  if ! python "$(resolve_path hpc.py)" shard "${stage}" --manifest "${manifest}"; then
    rc=$?
    err_msg="${err_msg:+$err_msg; }hpc.py shard failed (rc=${rc})"
  fi

  if on_worker; then
    condor_py package_shard_outputs.py "${stage}" --manifest "${manifest}" --error "${err_msg}" || {
      rc=$?
      echo "package_shard_outputs failed (rc=${rc})" >&2
    }
  fi

  return "${rc}"
}

run_merge() {
  local stage="$1"
  local rc=0
  if on_worker; then
    condor_py fetch_shards_for_merge.py "${stage}" || rc=$?
    mkdir -p data/processed data/features
  fi
  python "$(resolve_path hpc.py)" merge "${stage}" || rc=$?
  if on_worker; then
    condor_py publish_merge_outputs.py "${stage}" || rc=$?
  fi
  return "${rc}"
}

setup_python
if on_worker; then
  echo "=== worker: $(hostname) pwd=$(pwd) scratch=${_CONDOR_SCRATCH_DIR} ==="
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
    python "$(resolve_path hpc.py)" "$@"
    ;;
esac

echo "=== done ==="
