"""CRIT-02: pipeline provenance artifact for results/metrics drift detection."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.evaluation.reporter import ReportGenerator
from src.utils.pipeline_version import (
    PIPELINE_VERSION_FILENAME,
    build_pipeline_version_record,
    config_dict_hash,
    write_pipeline_version,
)


def _minimal_config(tmp_path: Path) -> dict:
    cfg_path = tmp_path / "pipeline_config.yaml"
    cfg = {
        "paths": {"metrics": str(tmp_path / "metrics")},
        "reproducibility": {"seed": 42},
        "models": {
            "evaluation": {
                "random_state": 42,
                "primary_endpoint": "anomaly_ensemble",
            }
        },
        "feature_selection": {
            "enabled": True,
            "primary_method": "rfecv",
            "rfecv_importance_method": "permutation",
            "required_feature_substrings": ["sampen", "dfa"],
        },
        "anomaly_detection": {"enabled": True},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg["_pipeline_meta"] = {"config_path": str(cfg_path.resolve())}
    return cfg


def test_config_dict_hash_stable():
    cfg = {"a": 1, "b": {"c": 2}}
    assert config_dict_hash(cfg) == config_dict_hash({"b": {"c": 2}, "a": 1})


def test_build_pipeline_version_includes_key_fields(tmp_path):
    cfg = _minimal_config(tmp_path)
    record = build_pipeline_version_record(cfg)
    assert record["pipeline_seed"] == 42
    assert record["primary_endpoint"] == "anomaly_ensemble"
    assert record["feature_selection"]["rfecv_importance_method"] == "permutation"
    assert record["config_sha256"]
    assert record["config_file_sha256"]
    assert "git" in record


def test_write_pipeline_version_creates_json(tmp_path):
    cfg = _minimal_config(tmp_path)
    metrics_dir = tmp_path / "metrics"
    out = write_pipeline_version(metrics_dir, cfg)
    assert out.name == PIPELINE_VERSION_FILENAME
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["feature_selection"]["required_feature_substrings"] == ["sampen", "dfa"]


def test_report_stage_writes_pipeline_version_first(tmp_path, monkeypatch):
    cfg = _minimal_config(tmp_path)
    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.csv").write_text(
        "model,auc,accuracy,f1,sensitivity,validation_strategy,participants\n"
        "xgboost,0.9,0.8,0.7,0.6,nested,10\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ReportGenerator,
        "_regenerate_demographics",
        lambda self: None,
    )
    monkeypatch.setattr(
        ReportGenerator,
        "_ensure_significance_pvalues",
        lambda self: None,
    )
    monkeypatch.setattr(
        ReportGenerator,
        "_generate_latex_table",
        lambda self, df: None,
    )
    monkeypatch.setattr(
        ReportGenerator,
        "_generate_markdown_report",
        lambda self, df: None,
    )

    ReportGenerator(cfg).run()
    assert (metrics_dir / PIPELINE_VERSION_FILENAME).is_file()
