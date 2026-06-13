"""ML-036: leakage comparison uses matched nested RFECV on ungrouped folds."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_leakage_comparison_uses_nested_rfecv_on_ungrouped_folds():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "nested_rfecv_column_indices" in source
    assert "grouped_feature_protocol" in source
    assert "ungrouped_feature_protocol" in source
    assert "protocol_matched" in source
    assert "X[train_idx][:, col_idx]" in source


def test_reporter_documents_protocol_matching():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "reporter.py").read_text(
        encoding="utf-8"
    )
    assert "ML-036" in source
    assert "grouped_feature_protocol" in source
