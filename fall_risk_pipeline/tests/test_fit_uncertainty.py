"""Tests for the fit_uncertainty pipeline stage (src/evaluation/fit_uncertainty.py).

This stage consumes the 'evaluate' stage's existing oof_predictions
artifact and fits calibration + conformal-prediction artifacts — it never
retrains or refits any upstream model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.fit_uncertainty import run_fit_uncertainty  # noqa: E402


def _synthetic_multiclass_oof(metrics_dir: Path, *, n=2000, k=3, seed=0, extra_model_rows=50) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) * 0.6
    for i in range(n):
        logits[i, y_true[i]] += rng.normal(2.0, 0.8)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    rows = []
    for i in range(n):
        row = {"model": "xgboost", "y_true": int(y_true[i]), "y_pred": int(probs[i].argmax())}
        for c in range(k):
            row[f"y_prob_class_{c}"] = float(probs[i, c])
        rows.append(row)
    for i in range(extra_model_rows):
        row = {"model": "svm", "y_true": int(y_true[i]), "y_pred": int(probs[i].argmax())}
        for c in range(k):
            row[f"y_prob_class_{c}"] = float(probs[i, c])
        rows.append(row)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "oof_predictions.csv", index=False)


def test_fit_uncertainty_end_to_end(tmp_path):
    metrics_dir = tmp_path / "metrics"
    _synthetic_multiclass_oof(metrics_dir)

    config = {
        "paths": {"metrics": str(metrics_dir)},
        "dataset": {"label_mode": "multiclass"},
        "reproducibility": {"seed": 42},
        "calibration": {"enabled": True, "conformal_alpha": 0.1},
    }
    report = run_fit_uncertainty(config)

    assert report["model"] == "xgboost"  # auto-selects the model with more OOF rows
    assert (metrics_dir / "calibration_artifact.json").is_file()
    assert (metrics_dir / "conformal_artifact.json").is_file()
    assert (metrics_dir / "uncertainty_coverage_report.json").is_file()

    cov = report["coverage_check"]
    assert cov["empirical_coverage"] >= cov["target_coverage"] - 0.05


def test_fit_uncertainty_explicit_model_selection(tmp_path):
    metrics_dir = tmp_path / "metrics"
    _synthetic_multiclass_oof(metrics_dir)
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "dataset": {"label_mode": "multiclass"},
        "calibration": {"enabled": True, "model": "svm"},
    }
    report = run_fit_uncertainty(config)
    assert report["model"] == "svm"


def test_fit_uncertainty_unknown_model_raises(tmp_path):
    metrics_dir = tmp_path / "metrics"
    _synthetic_multiclass_oof(metrics_dir)
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "dataset": {"label_mode": "multiclass"},
        "calibration": {"enabled": True, "model": "does_not_exist"},
    }
    with pytest.raises(ValueError):
        run_fit_uncertainty(config)


def test_fit_uncertainty_missing_oof_file_raises(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    config = {"paths": {"metrics": str(metrics_dir)}, "dataset": {"label_mode": "multiclass"}}
    with pytest.raises(FileNotFoundError):
        run_fit_uncertainty(config)


def test_fit_uncertainty_disabled_returns_empty(tmp_path):
    metrics_dir = tmp_path / "metrics"
    config = {"paths": {"metrics": str(metrics_dir)}, "calibration": {"enabled": False}}
    assert run_fit_uncertainty(config) == {}


def test_fit_uncertainty_artifacts_are_valid_json(tmp_path):
    metrics_dir = tmp_path / "metrics"
    _synthetic_multiclass_oof(metrics_dir)
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "dataset": {"label_mode": "multiclass"},
        "calibration": {"enabled": True},
    }
    run_fit_uncertainty(config)

    cal = json.loads((metrics_dir / "calibration_artifact.json").read_text())
    assert cal["label_mode"] == "multiclass"
    assert cal["n_classes"] == 3

    conf = json.loads((metrics_dir / "conformal_artifact.json").read_text())
    assert 0.0 <= conf["q_hat"] <= 1.0
    assert conf["alpha"] == 0.1
