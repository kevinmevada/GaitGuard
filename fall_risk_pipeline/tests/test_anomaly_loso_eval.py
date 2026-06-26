"""Tests for ANOM-001 LOSO anomaly evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _synthetic_trial_features(tmp_path: Path, n_features: int = 6) -> Path:
    rng = np.random.default_rng(42)
    rows = []
    feature_cols = [f"f{i}" for i in range(n_features)]
    pid = 0
    for cohort in ("Healthy", "Healthy", "PD", "HipOA"):
        for trial in range(8):
            row = {
                "trial_id": f"t_{pid}_{trial}",
                "participant_id": f"p{pid}",
                "cohort": cohort,
                "risk_label": 0 if cohort == "Healthy" else 1,
                "multiclass_label": 0 if cohort == "Healthy" else 2,
                "fall_probability": 0.05 if cohort == "Healthy" else 0.5,
                "laterality_biased": 0,
            }
            base = 0.0 if cohort == "Healthy" else 2.5
            for j, col in enumerate(feature_cols):
                row[col] = float(base + rng.normal(0, 0.3) + 0.1 * j)
            rows.append(row)
        pid += 1

    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(feat_dir / "trial_features.parquet", index=False)
    return feat_dir


def test_run_anomaly_loso_evaluation_exports_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.anomaly_loso_evaluator import run_anomaly_loso_evaluation

    feat_dir = _synthetic_trial_features(tmp_path)
    metrics_dir = tmp_path / "results" / "metrics"
    config = {
        "paths": {
            "features": str(feat_dir),
            "metrics": str(metrics_dir),
        },
        "reproducibility": {"seed": 42},
        "models": {"evaluation": {"primary_endpoint": "anomaly_ensemble"}},
    }

    metrics_df = run_anomaly_loso_evaluation(config)
    assert not metrics_df.empty
    assert "ensemble" in set(metrics_df["method"])
    assert (metrics_dir / "anomaly_metrics.csv").is_file()
    assert (metrics_dir / "anomaly_loso_oof_scores.csv").is_file()
    assert (metrics_dir / "primary_endpoint.json").is_file()

    reg = json.loads((metrics_dir / "primary_endpoint.json").read_text(encoding="utf-8"))
    assert reg["primary_endpoint"] == "anomaly_ensemble"
    assert "anomaly_ensemble" in reg["registered_endpoints"]


def test_anomaly_scoring_helpers():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.anomaly_scoring import (
        ANOMALY_METHODS,
        ensemble_scores,
        eval_binary_labels,
        fit_method_scores,
        normalise_scores,
    )

    rng = np.random.default_rng(0)
    X_h = rng.normal(size=(12, 4))
    X_q = rng.normal(size=(5, 4))
    for method in ANOMALY_METHODS:
        sq, sr, binary, _, _ = fit_method_scores(X_h, X_q, method, random_state=42)
        assert sq.shape == (5,)
        assert sr.shape == (12,)
        assert binary.shape == (5,)

    cohorts = np.array(["Healthy", "PD", "Healthy", "ACL"])
    assert list(eval_binary_labels(cohorts)) == [0, 1, 0, 1]

    scores = np.array([1.0, 2.0, 3.0, 4.0])
    normed = normalise_scores(scores, scores[:2])
    assert normed.min() >= 0.0 and normed.max() <= 1.0

    full = np.arange(8, dtype=float)
    ens = ensemble_scores(
        {"a": full, "b": full * 2},
        reference_masks={"a": np.arange(4), "b": np.arange(4)},
    )
    assert ens.shape == (8,)


def test_pipeline_config_primary_endpoint_anomaly():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    assert cfg["models"]["evaluation"]["primary_endpoint"] in (
        "anomaly_ensemble",
        "bilstm_ae_ensemble",
    )
    assert cfg["anomaly"]["loso_evaluation"] is True


def test_write_deploy_loso_skips_primary_json_when_anomaly(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.primary_endpoint import (
        ENDPOINT_ANOMALY_ENSEMBLE,
        write_deploy_loso_artifacts,
    )

    registry = {"primary_endpoint": ENDPOINT_ANOMALY_ENSEMBLE, "registered_endpoints": {}}
    write_deploy_loso_artifacts(tmp_path, [], registry)
    assert not (tmp_path / "primary_endpoint.json").exists()
