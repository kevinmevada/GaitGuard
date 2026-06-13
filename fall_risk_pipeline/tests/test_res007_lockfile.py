"""RES-007: requirements-lock.txt must pin all direct pipeline dependencies."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK = REPO_ROOT / "fall_risk_pipeline" / "requirements-lock.txt"
GENERATOR = REPO_ROOT / "scripts" / "generate_requirements_lock.py"

REQUIRED_PINS = {
    "pandas",
    "scipy",
    "scikit-learn",
    "numpy",
    "torch",
    "xgboost",
    "lightgbm",
    "pyarrow",
    "shap",
}


def test_lockfile_includes_core_packages():
    text = LOCK.read_text(encoding="utf-8").lower()
    for pkg in REQUIRED_PINS:
        assert f"{pkg}==" in text, f"missing pin for {pkg}"


def test_lockfile_uses_cpu_torch_not_cuda():
    text = LOCK.read_text(encoding="utf-8")
    assert "+cu128" not in text
    assert "torch==2." in text
    assert "+cpu" in text


def test_lockfile_generator_supports_torch_index_and_install():
    source = GENERATOR.read_text(encoding="utf-8")
    assert "--torch-index" in source
    assert "--install" in source
    assert "TORCH_CPU_INDEX" in source
    assert "TORCH_CUDA_INDEX" in source
    assert "pip_show_pin" in source
