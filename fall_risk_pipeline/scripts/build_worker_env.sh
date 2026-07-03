#!/bin/bash
# Build the prebuilt relocatable worker environment and stage it to OSDF.
#
# Run ONCE on ap40 (and again only when requirements change — bump VERSION
# here AND the WORKER_ENV macro in condor/hpc_shard_ingest.sub,
# condor/hpc_shard_cpu.sub, condor/hpc_merge.sub, because OSDF caches are
# immutable: never overwrite an existing tarball, always publish a new name).
#
# Usage (ap40):
#   cd ~/projects/GaitGuard/fall_risk_pipeline
#   bash scripts/build_worker_env.sh            # builds v1
#   bash scripts/build_worker_env.sh 2          # builds v2
set -euo pipefail

# Python 3.12: newest interpreter with cp312 wheels for every pin, and the
# minimum version where nolds 0.6.3 imports (its importlib.resources.files()
# module-anchor usage raises TypeError on <=3.11). 3.13 has no numpy 1.26.4 /
# numba 0.60 wheels, so 3.12 is the ceiling.
VERSION="${1:-1}"
PY_VERSION="3.12"
NAME="gg-worker-py312-v${VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING="${STAGING_LOCAL:-/ospool/ap40/data/kevin.mevada}"
MINIFORGE="${STAGING}/miniforge3"
DEST_DIR="${STAGING}/gaitguard/worker-env"
DEST="${DEST_DIR}/${NAME}.tar.gz"
BUILD_ENV="${STAGING}/tmp/${NAME}-build"

if [[ -f "${DEST}" ]]; then
  echo "ERROR: ${DEST} already exists. OSDF objects are cached immutably —"
  echo "       bump the version (bash scripts/build_worker_env.sh $((VERSION + 1)))"
  echo "       and update WORKER_ENV in the condor/*.sub files to match."
  exit 1
fi

export TMPDIR="${STAGING}/tmp"
mkdir -p "${TMPDIR}" "${DEST_DIR}"

# shellcheck source=/dev/null
source "${MINIFORGE}/bin/activate"

echo "=== [1/4] create build env (python ${PY_VERSION}) ==="
rm -rf "${BUILD_ENV}"
# Pin setuptools via conda, not pip — pip downgrading conda's setuptools leaves
# orphaned egg-info files and conda-pack refuses to pack the prefix.
conda create -y -p "${BUILD_ENV}" \
  "python=${PY_VERSION}" pip wheel "setuptools<81"

echo "=== [2/4] pip install pinned worker requirements (ingest + features) ==="
"${BUILD_ENV}/bin/pip" install --no-cache-dir \
  -r "${PROJECT}/condor/requirements-hpc-ingest.txt" \
  -r "${PROJECT}/condor/requirements-hpc-cpu.txt"

# Pip must not own setuptools — re-sync conda metadata before conda-pack.
echo "=== [2b/4] re-sync setuptools (conda-owned) ==="
conda install -y -p "${BUILD_ENV}" -c conda-forge --force-reinstall "setuptools<81"

echo "=== [3/4] verify imports ==="
"${BUILD_ENV}/bin/python" - <<'PY'
import importlib
mods = ["pandas", "numpy", "pyarrow", "scipy", "sklearn", "yaml", "loguru",
        "tqdm", "ahrs", "antropy", "nolds", "pywt", "numba"]
for m in mods:
    importlib.import_module(m)
    print(f"OK {m}")
import sys
print("python", sys.version)
PY

echo "=== [4/4] conda-pack -> ${DEST} ==="
if ! command -v conda-pack >/dev/null 2>&1; then
  pip install --no-cache-dir conda-pack
fi
# Use the standalone conda-pack executable, not "conda pack" — the latter needs
# conda-pack registered as a conda plugin, which it is not under miniforge here.
conda-pack -p "${BUILD_ENV}" -o "${DEST}" --compress-level 5

rm -rf "${BUILD_ENV}"
ls -lh "${DEST}"
echo ""
echo "Done. Submit files reference: \$(OSDF_GAITGUARD)/worker-env/${NAME}.tar.gz"
echo "Smoke-test the tarball locally on ap40:"
echo "  mkdir -p /tmp/ggenv && tar -xzf ${DEST} -C /tmp/ggenv && /tmp/ggenv/bin/python -c 'import pandas; print(\"ok\")' && rm -rf /tmp/ggenv"
