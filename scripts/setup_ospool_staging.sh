#!/bin/bash
# One-time OSPool staging layout for GaitGuard (run on ap40 login node).
set -euo pipefail

STAGING="/ospool/ap40/data/kevin.mevada"
PROJECT="${HOME}/projects/GaitGuard/fall_risk_pipeline"

mkdir -p "${STAGING}/gaitguard"/{raw,processed,features,results,hpc/shards,hpc/oof,hpc/merge}
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

cd "${PROJECT}/data"
if [ -L hpc ]; then
  echo "Removing legacy data/hpc -> OSDF symlink (manifests must live on AP for condor transfer)"
  rm hpc
fi
mkdir -p hpc/manifests
for sub in shards oof merge; do
  target="${STAGING}/gaitguard/hpc/${sub}"
  link="hpc/${sub}"
  if [ -L "${link}" ]; then
    echo "OK: ${link} already symlinked"
  elif [ -e "${link}" ]; then
    echo "WARN: data/${link} exists and is not a symlink"
  else
    ln -s "${target}" "${link}"
    echo "linked ${link} -> ${target}"
  fi
done

mkdir -p "${PROJECT}/condor/manifests"

echo "Staging ready under ${STAGING}/gaitguard"
