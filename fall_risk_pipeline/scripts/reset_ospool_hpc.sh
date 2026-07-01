#!/bin/bash
# Wipe failed/partial HPC runs and submit a fresh sharded DAG (ap40 login node).
#
# Keeps: raw data, conda env, dataset_inventory.csv (from discover).
# Clears: condor jobs, logs, rescue DAGs, OSDF shard/merge/oof, processed parquets,
#         features, results, and regenerates manifests + DAG.
#
# Usage:
#   bash scripts/reset_ospool_hpc.sh           # interactive confirm
#   bash scripts/reset_ospool_hpc.sh --yes     # no prompt
set -euo pipefail

YES=false
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && YES=true

STAGING="/ospool/ap40/data/kevin.mevada"
GG="${STAGING}/gaitguard"
CONDA_ENV="${STAGING}/miniforge3/envs/gaitguard"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"
OWNER="$(whoami)"

confirm() {
  if $YES; then return 0; fi
  echo "This will:"
  echo "  - git pull latest code"
  echo "  - condor_rm ALL jobs for ${OWNER}"
  echo "  - delete condor/logs/* and condor/dags/*.{rescue*,condor.sub,dagman.*,lib.*}"
  echo "  - delete ${GG}/hpc/{shards,merge,oof}/*"
  echo "  - delete processed parquets (keep dataset_inventory.csv)"
  echo "  - delete ${GG}/features/* and ${GG}/results/*"
  echo "  - regenerate manifests + DAG and condor_submit_dag"
  read -r -p "Continue? [y/N] " ans
  [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]]
}

confirm || { echo "Aborted."; exit 0; }

echo "=== 0. Pull latest code ==="
cd "${HOME}/projects/GaitGuard"
git pull
cd "${PROJECT}"

echo "=== 1. Remove all HTCondor jobs ==="
condor_rm "${OWNER}" 2>/dev/null || true
sleep 2
remaining="$(condor_q "${OWNER}" 2>/dev/null | tail -1 | awk '{print $1}')"
echo "Queue after rm: ${remaining:-empty}"

echo "=== 2. Clean local condor artifacts ==="
cd "${PROJECT}"
rm -rf condor/logs/*
rm -f condor/dags/*.rescue* \
      condor/dags/*.condor.sub \
      condor/dags/*.dagman.* \
      condor/dags/*.lib.*

echo "=== 3. Clean OSDF partial pipeline outputs ==="
rm -rf "${GG}/hpc/shards"/* "${GG}/hpc/merge"/* "${GG}/hpc/oof"/* 2>/dev/null || true
mkdir -p "${GG}/hpc"/{shards,merge,oof}

# Processed: keep discover inventory only
if [[ -d "${GG}/processed" ]]; then
  find "${GG}/processed" -mindepth 1 -maxdepth 1 ! -name 'dataset_inventory.csv' -exec rm -rf {} +
fi
rm -rf "${GG}/features"/* "${GG}/results"/* 2>/dev/null || true
mkdir -p "${GG}/features" "${GG}/results"

echo "=== 4. Regenerate manifests ==="
# shellcheck source=/dev/null
source "${STAGING}/miniforge3/bin/activate" "${CONDA_ENV}"
export TMPDIR="${STAGING}/tmp"
mkdir -p "${TMPDIR}"
cd "${PROJECT}"

echo "  python: $(which python) ($(python --version 2>&1))"
rm -f condor/manifests/*.json

echo "  running: hpc.py init ..."
python hpc.py init
echo "  running: hpc.py manifests ingest ..."
python hpc.py manifests ingest
echo "  running: hpc.py manifests preprocess ..."
python hpc.py manifests preprocess
echo "  running: hpc.py manifests features ..."
python hpc.py manifests features
echo "  manifests: $(ls condor/manifests/*.json 2>/dev/null | wc -l) files"

echo "=== 5. OSDF OAuth token (required for stashcp on workers) ==="
if ! condor_store_cred query-oauth -s scitokens >/dev/null 2>&1; then
  echo "No scitokens credential stored. Run this ONCE (browser login), then re-run this script:"
  echo "  condor_vault_storer -v scitokens"
  exit 1
fi
echo "  scitokens credential OK"

echo "=== 6. Regenerate DAG and submit ==="
python hpc/submit/generate_dag.py --config configs/pipeline_config.yaml
condor_submit_dag -f condor/dags/gaitguard_sharded.dag

echo ""
echo "=== Done. Monitor with: ==="
echo "  condor_q -dag ${OWNER}"
echo "  ls ${GG}/hpc/shards/ingest/*.tar.gz 2>/dev/null | wc -l   # target 68"
