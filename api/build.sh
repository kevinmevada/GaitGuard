#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install --prefer-binary -r requirements-lock.txt
python ../scripts/sync_front_end.py

if [ -n "${GAITGUARD_HF_REPO:-}" ]; then
  if [ "${ENVIRONMENT:-}" = "production" ] || [ "${ENVIRONMENT:-}" = "prod" ]; then
    if [ -z "${CHECKPOINT_HMAC_KEY:-}" ]; then
      echo "ERROR: CHECKPOINT_HMAC_KEY must be set for production builds (SEC-009)." >&2
      exit 1
    fi
    if [ -z "${GAITGUARD_HF_REVISION:-}" ] || [ "${GAITGUARD_HF_REVISION}" = "main" ]; then
      echo "ERROR: Pin GAITGUARD_HF_REVISION to an immutable Hub tag (SEC-009)." >&2
      exit 1
    fi
  fi
  pip install huggingface_hub
  python ../scripts/download_models.py
else
  echo "ERROR: GAITGUARD_HF_REPO must be set in Render to download inference models." >&2
  exit 1
fi
