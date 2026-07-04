#!/usr/bin/env bash
# One-time local setup for GaitGuard (Linux/macOS/Git Bash).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON=python
fi

echo "=== [1/4] Create virtual environment ==="
if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

echo "=== [2/4] Install dependencies ==="
pip install --upgrade pip wheel
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

echo "=== [3/4] Create local data directories ==="
mkdir -p \
  fall_risk_pipeline/data/raw \
  fall_risk_pipeline/data/raw_local \
  fall_risk_pipeline/data/processed \
  fall_risk_pipeline/data/features \
  fall_risk_pipeline/logs \
  fall_risk_pipeline/results/metrics \
  fall_risk_pipeline/results/figures \
  fall_risk_pipeline/results/checkpoints

echo "=== [4/4] Verify import ==="
export PYTHONHASHSEED=42
cd fall_risk_pipeline
python -c "import pandas, numpy, torch, yaml; print('OK', __import__('sys').version)"

echo ""
echo "Setup complete. Run the full pipeline:"
echo "  source .venv/bin/activate"
echo "  export PYTHONHASHSEED=42"
echo "  python run_local.py"
echo ""
echo "Smoke test with synthetic data:"
echo "  python run_local.py --use-local-config --seed-data --trials 6"
