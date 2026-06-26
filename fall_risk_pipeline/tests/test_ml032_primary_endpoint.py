"""Tests for ML-032 deploy vs nested LOSO feature-schema gap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_pipeline_config_registers_primary_endpoint():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    ev = cfg["models"]["evaluation"]
    assert ev.get("primary_endpoint") in ("anomaly_ensemble", "bilstm_ae_ensemble")
    assert ev.get("report_deploy_loso_gap") is True


def test_evaluator_exports_deploy_loso_gap():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "_evaluate_deploy_schema_loso" in source
    assert "deploy_loso_gap.csv" in source
    assert "apply_nested_fs" in source


def test_predictions_uses_primary_deploy_checkpoint():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "predictions.py").read_text(
        encoding="utf-8"
    )
    assert "resolve_inference_checkpoint_name" in source
    assert "Loaded primary deploy checkpoint" in source


def test_build_deploy_loso_gap_rows():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.primary_endpoint import (
        ENDPOINT_DEPLOY_ENSEMBLE,
        build_deploy_loso_gap_rows,
        build_primary_endpoint_registry,
    )

    nested = {
        "random_forest": {"auc": 0.90, "accuracy": 0.85},
        "ensemble_soft_voting": {"auc": 0.88, "accuracy": 0.83},
    }
    deploy = {
        "random_forest": {"auc": 0.87, "accuracy": 0.82},
        "ensemble_soft_voting": {"auc": 0.86, "accuracy": 0.81},
    }
    rows = build_deploy_loso_gap_rows(
        nested,
        deploy,
        nested_feature_count_median=18,
        deploy_feature_count=20,
    )
    assert len(rows) == 2
    rf = next(r for r in rows if r["model"] == "random_forest")
    assert rf["loso_auc_nested_rfecv"] == 0.90
    assert rf["loso_auc_deploy_schema"] == 0.87
    assert abs(rf["delta_auc_deploy_minus_nested"] - (-0.03)) < 1e-9
    assert rf["nested_feature_count_median"] == 18
    assert rf["deploy_feature_count"] == 20

    config = {
        "models": {
            "evaluation": {"primary_endpoint": "deploy_ensemble"},
            "ensemble": {"method": "soft_voting"},
        }
    }
    registry = build_primary_endpoint_registry(config, nested, deploy)
    assert registry["primary_endpoint"] == ENDPOINT_DEPLOY_ENSEMBLE
    assert registry["registered_endpoints"]["deploy_ensemble"]["model"] == "ensemble_soft_voting"
    assert registry["registered_endpoints"]["best_loso_nested"]["model"] == "random_forest"


def test_write_deploy_loso_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.primary_endpoint import write_deploy_loso_artifacts

    rows = [
        {
            "model": "xgboost",
            "loso_auc_nested_rfecv": 0.91,
            "loso_auc_deploy_schema": 0.89,
            "delta_auc_deploy_minus_nested": -0.02,
        }
    ]
    registry = {"primary_endpoint": "deploy_ensemble", "registered_endpoints": {}}
    write_deploy_loso_artifacts(tmp_path, rows, registry)

    gap = (tmp_path / "deploy_loso_gap.csv").read_text(encoding="utf-8")
    assert "loso_auc_nested_rfecv" in gap
    assert "0.91" in gap
    ep = json.loads((tmp_path / "primary_endpoint.json").read_text(encoding="utf-8"))
    assert ep["primary_endpoint"] == "deploy_ensemble"
