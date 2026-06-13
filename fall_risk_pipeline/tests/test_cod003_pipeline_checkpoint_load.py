"""COD-003: research pipeline loaders use load_checkpoint."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluator_loads_checkpoints_via_manifest_helper():
    source = (REPO_ROOT / "fall_risk_pipeline" / "src" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "load_checkpoint" in source
    assert "pickle.load" not in source


def test_ablation_modules_use_load_checkpoint():
    for rel in (
        "fall_risk_pipeline/src/evaluation/feature_ablation.py",
        "fall_risk_pipeline/src/evaluation/sensor_ablation.py",
    ):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "load_checkpoint" in source
        assert "pickle.load" not in source
