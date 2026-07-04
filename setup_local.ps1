# One-time local setup for GaitGuard (Windows PowerShell).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

Write-Host "=== [1/4] Create virtual environment ==="
if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

Write-Host "=== [2/4] Install dependencies ==="
pip install --upgrade pip wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

Write-Host "=== [3/4] Create local data directories ==="
$dirs = @(
    "fall_risk_pipeline/data/raw",
    "fall_risk_pipeline/data/raw_local",
    "fall_risk_pipeline/data/processed",
    "fall_risk_pipeline/data/features",
    "fall_risk_pipeline/logs",
    "fall_risk_pipeline/results/metrics",
    "fall_risk_pipeline/results/figures",
    "fall_risk_pipeline/results/checkpoints"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-Host "=== [4/4] Verify import ==="
$env:PYTHONHASHSEED = "42"
Set-Location fall_risk_pipeline
python -c "import pandas, numpy, torch, yaml; print('OK', __import__('sys').version)"

Write-Host ""
Write-Host "Setup complete. Run the full pipeline:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  `$env:PYTHONHASHSEED = '42'"
Write-Host "  python run_local.py"
