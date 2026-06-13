"""ML-040: forced nonlinear features ranked by SHAP; audit export for post-rerun review."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_config_documents_shap_ranked_required_cap():
    cfg = yaml.safe_load(
        (REPO_ROOT / "fall_risk_pipeline" / "configs" / "pipeline_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    fs = cfg["feature_selection"]
    assert int(fs.get("max_required_features", 99)) <= 4
    comment_block = (
        REPO_ROOT / "fall_risk_pipeline" / "configs" / "pipeline_config.yaml"
    ).read_text(encoding="utf-8")
    assert "ML-040" in comment_block or "mean |SHAP|" in comment_block


def test_feature_selector_exports_required_shap_audit():
    source = (
        REPO_ROOT / "fall_risk_pipeline" / "src" / "features" / "feature_selector.py"
    ).read_text(encoding="utf-8")
    assert "_rank_required_candidates" in source
    assert "required_feature_shap_audit.csv" in source
    assert "full_mean_abs_shap" in source


def test_limitations_note_forced_nonlinear_slots():
    text = (REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    assert "required_feature_shap_audit.csv" in text
    assert "max_required_features" in text
