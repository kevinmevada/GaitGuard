#!/usr/bin/env bash
# Simulate one HTCondor worker job locally (Git Bash / Linux / macOS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT}"

CONFIG="${GAITGUARD_CONFIG:-.local_staging/merged_pipeline_config.yaml}"
STAGING_ROOT="${PROJECT}/.local_staging"
GG="${STAGING_ROOT}/gaitguard"
SCRATCH="$(mktemp -d 2>/dev/null || mktemp -d -t gg_worker)"

cleanup() {
  if [[ "${KEEP_SCRATCH:-0}" != "1" ]]; then
    rm -rf "${SCRATCH}"
  else
    echo "KEEP_SCRATCH=1: ${SCRATCH}"
  fi
}
trap cleanup EXIT

CMD="${1:-}"
STAGE="${2:-ingest}"
if [[ "${CMD}" == "shard" || "${CMD}" == "merge" || "${CMD}" == "reduce" ]]; then
  export HPC_WORKER_STAGE="${STAGE}"
else
  export HPC_WORKER_STAGE="${HPC_WORKER_STAGE:-ingest}"
fi

export _CONDOR_SCRATCH_DIR="${SCRATCH}"
export GAITGUARD_LOCAL_STAGING="${GG}"
export STAGING_LOCAL="${STAGING_ROOT}"
export GAITGUARD_PROJECT="${PROJECT}"
# Relative config path so run_hpc.sh resolves the copy on scratch/configs/ and
# src/hpc/paths.pipeline_root() points at scratch — matching OSPool exactly.
export GAITGUARD_CONFIG="configs/pipeline_config.yaml"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PIP_DEFAULT_TIMEOUT=180
export PIP_PREFER_BINARY=1

mkdir -p "${GG}/raw" "${GG}/hpc/shards/ingest" "${GG}/hpc/shards/preprocess" \
  "${GG}/hpc/shards/features" "${GG}/hpc/merge_bundles" "${SCRATCH}/tmp" \
  "${SCRATCH}/configs" "${SCRATCH}/src"

copy_if() { [[ -e "$1" ]] && cp -r "$1" "$2"; }

# Mirror HTCondor transfer_input_files on scratch.
for f in run_hpc.sh ospool_paths.sh bootstrap_worker_env.sh local_paths.py \
  requirements-hpc-ingest.txt requirements-hpc-cpu.txt \
  stage_shard_inputs.py package_shard_outputs.py \
  fetch_shards_for_merge.py publish_merge_outputs.py; do
  copy_if "${PROJECT}/condor/${f}" "${SCRATCH}/${f}"
done
copy_if "${PROJECT}/hpc.py" "${SCRATCH}/hpc.py"
cp -r "${PROJECT}/configs/." "${SCRATCH}/configs/"
cp -r "${PROJECT}/src/." "${SCRATCH}/src/"
if [[ -f "${CONFIG}" ]]; then
  cp "${CONFIG}" "${SCRATCH}/configs/pipeline_config.yaml"
fi

if [[ "${CMD}" == "shard" && "${4:-}" == "--manifest" && -n "${5:-}" ]]; then
  man="${5}"
  if [[ -f "${PROJECT}/${man}" ]]; then
    cp "${PROJECT}/${man}" "${SCRATCH}/$(basename "${man}")"
  fi
fi

if [[ -d "${PROJECT}/data/raw_local" ]]; then
  mkdir -p "${GG}/raw"
  cp -r "${PROJECT}/data/raw_local/." "${GG}/raw/"
fi

pick_python() {
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
PY="$(pick_python || true)"
if [[ -z "${PY}" ]]; then
  echo "ERROR: need Python 3.10-3.12 to build worker venv" >&2
  exit 1
fi
VENV_TAG="ingest"
REQ="requirements-hpc-ingest.txt"
if [[ "${HPC_WORKER_STAGE}" == "features" ]]; then
  VENV_TAG="features"
  REQ="requirements-hpc-cpu.txt"
fi
# Persistent venv across local worker runs (OSPool builds per-scratch; locally
# rebuilding per shard wastes ~4 min of pip each time).
VENV="${STAGING_ROOT}/.worker-venv-${VENV_TAG}"

if [[ ! -x "${VENV}/bin/python" && ! -x "${VENV}/Scripts/python.exe" ]]; then
  echo "=== creating worker venv (${VENV_TAG}) ==="
  "${PY}" -m venv "${VENV}"
  PYBIN="${VENV}/bin/python"
  [[ -x "${PYBIN}" ]] || PYBIN="${VENV}/Scripts/python.exe"
  "${PYBIN}" -m pip install --upgrade pip wheel "setuptools<81"
  "${PYBIN}" -m pip install --no-cache-dir --only-binary=pyarrow -r "${SCRATCH}/${REQ}"
fi

# Windows venvs use Scripts/; run_hpc/bootstrap expect bin/.
if [[ -x "${VENV}/Scripts/python.exe" ]]; then
  mkdir -p "${VENV}/bin"
  cp -f "${VENV}/Scripts/python.exe" "${VENV}/bin/python"
  if [[ -x "${VENV}/Scripts/pip.exe" ]]; then
    cp -f "${VENV}/Scripts/pip.exe" "${VENV}/bin/pip"
  fi
fi
export WORKER_VENV="${VENV}"

echo "=== START local worker ==="
echo "hostname=$(hostname 2>/dev/null || echo local)"
echo "pid=$$"
echo "scratch=${SCRATCH}"
echo "config=${CONFIG}"
PYBIN="${VENV}/bin/python"
[[ -x "${PYBIN}" ]] || PYBIN="${VENV}/Scripts/python.exe"
echo "python=$("${PYBIN}" --version 2>&1)"
echo "args=$*"

t0=$(date +%s)
set +e
bash "${SCRATCH}/run_hpc.sh" "$@"
rc=$?
set -e
t1=$(date +%s)
echo "=== END local worker rc=${rc} elapsed=$((t1 - t0))s ==="

if [[ "${CMD}" == "shard" && "${rc}" -eq 0 && -f "${SCRATCH}/shard_out.tar.gz" ]]; then
  stage="${2:-}"
  chunk="$(basename "${5:-chunk_0000.json}" .json | sed 's/^ingest_//; s/^preprocess_//; s/^features_//')"
  dest="${PROJECT}/data/hpc/shards/${stage}/${chunk}"
  mkdir -p "${dest}"
  tar -xzf "${SCRATCH}/shard_out.tar.gz" -C "${SCRATCH}" 2>/dev/null || true
  if [[ -d "${SCRATCH}/data/hpc/shards/${stage}/${chunk}" ]]; then
    cp -r "${SCRATCH}/data/hpc/shards/${stage}/${chunk}/." "${dest}/"
    echo "shard unpacked -> ${dest}"
  fi
fi

exit "${rc}"
