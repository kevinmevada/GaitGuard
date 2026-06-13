"""Tests for ML-033 feature ablation SHAP / LOSO schema alignment."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_loso_shap_uses_nested_rfecv_intersect():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "feature_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "intersect_nested_rfecv_columns" in source
    assert "loso_aggregate_nested_rfecv" in source
    assert "X[train_idx][:, fold_cols]" in source
    assert "X[test_idx][:, fold_cols]" in source


def test_baseline_scenario_uses_nested_rfecv_name():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "feature_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "all_features_nested_rfecv" in source
    assert 'BASELINE_ABLATION_SCENARIO' in source
    assert "no RFECV mask" not in source
    assert '"all_features"' not in source or "all_features_nested_rfecv" in source
    tree = ast.parse(
        (PIPELINE_ROOT / "src" / "evaluation" / "feature_ablation.py").read_text(
            encoding="utf-8"
        )
    )
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_loso_shap_importance"
    )
    body = ast.get_source_segment(
        (PIPELINE_ROOT / "src" / "evaluation" / "feature_ablation.py").read_text(
            encoding="utf-8"
        ),
        fn,
    )
    assert body is not None
    assert "fold_cols = intersect_nested_rfecv_columns" in body
    assert "feat_names[col_idx]" in body
