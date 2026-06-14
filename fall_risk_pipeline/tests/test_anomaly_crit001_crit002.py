"""CRIT-001/002: anomaly LOSO train-fold Youden and in-sample artifact labeling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_anomaly_loso_uses_train_fold_youden_not_oof(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.anomaly_loso_evaluator import run_anomaly_loso_evaluation

    source = (PIPELINE_ROOT / "src" / "evaluation" / "anomaly_loso_evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "youden_threshold(y_train, sq_train)" in source
    assert "loso_train_fold_youden" in source
    score_block = source.split("def _score_block", 1)[1].split("\ndef run_anomaly", 1)[0]
    assert "youden_threshold(yt, ys)" not in score_block

    feat_dir = _synthetic_trial_features(tmp_path)
    metrics_dir = tmp_path / "results" / "metrics"
    config = {
        "paths": {"features": str(feat_dir), "metrics": str(metrics_dir)},
        "reproducibility": {"seed": 42},
        "models": {"evaluation": {"primary_endpoint": "anomaly_ensemble"}},
    }

    metrics_df = run_anomaly_loso_evaluation(config)
    ens = metrics_df.loc[metrics_df["method"] == "ensemble"].iloc[0]
    assert ens["threshold_source"] == "loso_train_fold_youden"
    assert ens["n_threshold_folds"] >= 1
    assert pd.notna(ens["threshold_youden"])

    oof = pd.read_csv(metrics_dir / "anomaly_loso_oof_scores.csv")
    assert "ensemble_pred_train_youden" in oof.columns
    assert oof["ensemble_pred_train_youden"].notna().sum() > 0

    thresh_json = (metrics_dir / "anomaly_threshold.json").read_text(encoding="utf-8")
    assert "loso_train_fold_youden_mean" in thresh_json


def test_score_fitted_method_matches_dual_fit():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.anomaly_scoring import fit_method_scores, score_fitted_method

    rng = np.random.default_rng(1)
    X_h = rng.normal(size=(14, 5))
    X_train = rng.normal(size=(20, 5))
    sq_test, _, _, model, scaler = fit_method_scores(
        X_h, X_train[:5], "isolation_forest", random_state=0
    )
    sq_train = score_fitted_method(model, scaler, X_train)
    sq_test2, _, _, _, _ = fit_method_scores(
        X_h, X_train[:5], "isolation_forest", random_state=0
    )
    assert np.allclose(sq_test, sq_test2)
    assert sq_train.shape == (20,)


def test_bulk_anomaly_outputs_labeled_insample(tmp_path, monkeypatch):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.anomaly_detector import GaitAnomalyDetector, INSAMPLE_ARTIFACT_BANNER

    feat_dir = _synthetic_trial_features(tmp_path)
    results_dir = tmp_path / "results" / "anomaly_detection"
    config = {
        "paths": {
            "features": str(feat_dir),
            "results": str(tmp_path / "results"),
            "metrics": str(tmp_path / "results" / "metrics"),
        },
        "reproducibility": {"seed": 42},
        "anomaly": {"loso_evaluation": False},
        "models": {"evaluation": {"primary_endpoint": "anomaly_ensemble"}},
    }

    monkeypatch.setattr(
        "src.models.anomaly_detector.GaitAnomalyDetector._visualize_results",
        lambda *args, **kwargs: None,
    )

    det = GaitAnomalyDetector(config)
    det.run()

    csv_path = results_dir / "anomaly_exploratory_insample_ensemble_results.csv"
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert text.startswith(INSAMPLE_ARTIFACT_BANNER.split("\n")[0])
    assert not (results_dir / "ensemble_results.csv").exists()

    cohort_path = results_dir / "anomaly_exploratory_insample_cohort_analysis.json"
    assert cohort_path.is_file()
    assert "_disclaimer" in cohort_path.read_text(encoding="utf-8")
