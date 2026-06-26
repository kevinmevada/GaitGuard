"""Tests for core discriminative competitor metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_binary_discriminative_metrics_perfect():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.competitor_metrics import compute_discriminative_metrics

    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_discriminative_metrics(
        y, pred, y_score=score, config={"dataset": {"label_mode": "binary"}}
    )
    assert m["f1_weighted"] == 1.0
    assert m["mcc"] == 1.0
    assert m["auroc"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["cohen_kappa"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["precision"] == 1.0


def test_multiclass_metrics_keys():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.competitor_metrics import compute_discriminative_metrics

    y = np.array([0, 0, 1, 2, 2])
    pred = np.array([0, 1, 1, 2, 2])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.4, 0.4, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
        ]
    )
    m = compute_discriminative_metrics(
        y, pred, y_proba=proba, config={"dataset": {"label_mode": "multiclass"}}
    )
    for key in ("f1_weighted", "balanced_accuracy", "mcc", "auroc", "cohen_kappa"):
        assert key in m
        assert np.isfinite(m[key])


def test_competitor_matrix_aggregator(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.competitor_matrix_aggregator import run_competitor_discriminative_matrix

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd = __import__("pandas")
    pd.DataFrame(
        [
            {
                "model": "svm_rbf",
                "f1_weighted": 0.7,
                "balanced_accuracy": 0.68,
                "mcc": 0.55,
                "auroc": 0.82,
                "sensitivity": 0.71,
                "specificity": 0.65,
                "precision": 0.69,
                "cohen_kappa": 0.5,
            }
        ]
    ).to_csv(metrics_dir / "classical_baseline_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "bilstm_ae_ensemble",
                "display_name": "BiLSTM-AE",
                "f1_weighted": 0.75,
                "balanced_accuracy": 0.72,
                "mcc": 0.71,
                "auroc": 0.88,
                "sensitivity": 0.74,
                "specificity": 0.7,
                "precision": 0.73,
                "cohen_kappa": 0.68,
                "paradigm": "gaitguard_primary",
            }
        ]
    ).to_csv(metrics_dir / "dl_baseline_metrics.csv", index=False)

    config = {
        "paths": {"metrics": str(metrics_dir)},
        "competitor_metrics": {"enabled": True, "mcc_abstract_lead_threshold": 0.7},
    }
    df = run_competitor_discriminative_matrix(config)
    assert len(df) >= 2
    assert (metrics_dir / "competitor_discriminative_metrics.csv").is_file()
    summary = __import__("json").loads(
        (metrics_dir / "competitor_discriminative_summary.json").read_text(encoding="utf-8")
    )
    assert summary["abstract_headline"] == "mcc_primary"
