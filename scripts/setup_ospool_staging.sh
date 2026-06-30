#!/bin/bash
# One-time OSPool staging layout for GaitGuard (run on ap40 login node).
set -euo pipefail

STAGING="/ospool/ap40/data/kevin.mevada"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"

mkdir -p "${STAGING}/gaitguard"/{raw,processed,features,results}
mkdir -p "${STAGING}"/{conda/pkgs,conda/envs,pip-cache,tmp,cache}

cd "${PROJECT}/data"
for d in raw processed features; do
  target="${STAGING}/gaitguard/${d}"
  if [ -L "$d" ]; then
    echo "OK: $d already symlinked"
  elif [ -e "$d" ]; then
    echo "WARN: $d exists and is not a symlink — move aside manually then re-run"
  else
    ln -s "${target}" "$d"
    echo "linked $d -> ${target}"
  fi
done

cd "${PROJECT}"
if [ -L results ]; then
  echo "OK: results already symlinked"
elif [ -e results ] && [ ! -L results ]; then
  echo "WARN: results/ is a real directory — consider: mv results ${STAGING}/gaitguard/results_legacy && ln -s ${STAGING}/gaitguard/results results"
else
  ln -s "${STAGING}/gaitguard/results" results
  echo "linked results -> ${STAGING}/gaitguard/results"
fi

echo "Staging ready under ${STAGING}/gaitguard"
