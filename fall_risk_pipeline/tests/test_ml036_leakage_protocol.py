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
    assert "_ungrouped_kfold_auc" in source
    assert "leakage_kfold_seed_repeats_by_model" in source or "_leakage_kfold_seed_repeats" in source


def test_pipeline_config_averages_mlp_leakage_over_seeds():
    import yaml

    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    eval_cfg = cfg["models"]["evaluation"]
    assert int(eval_cfg.get("leakage_kfold_seed_repeats", 1)) >= 5
    by_model = eval_cfg.get("leakage_kfold_seed_repeats_by_model", {})
    assert int(by_model.get("mlp", 1)) >= 5


def test_limitations_documents_split_protocol_seed_averaging():
    text = (REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    assert "Split-protocol comparison" in text
    assert "leakage_kfold_seed_repeats" in text
    assert "split_protocol_comparison.csv" in text


def test_reporter_documents_protocol_matching():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "reporter.py").read_text(
        encoding="utf-8"
    )
    assert "ML-036" in source
    assert "grouped_feature_protocol" in source
