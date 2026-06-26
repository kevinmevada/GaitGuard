"""Tests for fall-risk Spearman clinical validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _synthetic_oof(tmp_path: Path) -> Path:
    """Monotone anomaly scores aligned with literature fall-risk ordering."""
    rows = []
    spec = [
        ("Healthy", 0.052, 8, 0.12),
        ("PD", 0.673, 6, 0.72),
        ("CVA", 0.542, 6, 0.58),
        ("HipOA", 0.285, 6, 0.35),
        ("KneeOA", 0.241, 6, 0.30),
        ("ACL", 0.187, 6, 0.25),
        ("CIPN", 0.418, 4, 0.48),
        ("RIL", 0.389, 4, 0.45),
    ]
    tid = 0
    rng = np.random.default_rng(1)
    for cohort, fp, n_trials, base in spec:
        for p in range(2):
            pid = f"{cohort[:2]}_{p}"
            for _ in range(n_trials // 2):
                score = float(base + rng.normal(0, 0.03))
                rows.append(
                    {
                        "trial_id": f"t{tid}",
                        "participant_id": pid,
                        "cohort": cohort,
                        "fall_probability": fp,
                        "eval_non_healthy": 0 if cohort == "Healthy" else 1,
                        "bilstm_ae_ensemble_score": score,
                    }
                )
                tid += 1

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    path = metrics_dir / "bilstm_ae_loso_oof_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return metrics_dir


def test_spearman_metrics_positive_global_correlation():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.fall_risk_spearman_metrics import compute_fall_risk_spearman_table

    rng = np.random.default_rng(0)
    rows = []
    for cohort, fp in [("Healthy", 5.2), ("PD", 67.3), ("HipOA", 28.5)]:
        for i in range(5):
            rows.append(
                {
                    "participant_id": f"{cohort}_{i}",
                    "cohort": cohort,
                    "fall_probability": fp / 100.0,
                    "bilstm_ae_ensemble_score": fp / 100.0 + rng.normal(0, 0.01),
                }
            )
    trial_df = pd.DataFrame(rows)
    summary, participants = compute_fall_risk_spearman_table(
        trial_df, "bilstm_ae_ensemble_score"
    )
    global_row = summary.loc[summary["comparison_scope"] == "global_all_participants"].iloc[0]
    assert global_row["defined"]
    assert float(global_row["spearman_rho"]) > 0.9
    assert float(global_row["p_value"]) < 0.05
    assert len(participants) == 15


def test_within_cohort_spearman_undefined():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.fall_risk_spearman_metrics import compute_fall_risk_spearman_table

    trial_df = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "cohort": ["PD", "PD", "PD"],
            "fall_probability": [0.673, 0.673, 0.673],
            "bilstm_ae_ensemble_score": [0.5, 0.6, 0.7],
        }
    )
    summary, _ = compute_fall_risk_spearman_table(trial_df, "bilstm_ae_ensemble_score")
    within = summary.loc[summary["comparison_scope"] == "within_PD"].iloc[0]
    assert not within["defined"]
    assert "constant" in str(within["note"]).lower()


def test_run_fall_risk_spearman_exports_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.fall_risk_spearman_evaluator import run_fall_risk_spearman_correlation

    metrics_dir = _synthetic_oof(tmp_path)
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "fall_risk_spearman": {"enabled": True, "sync_paper_docs": False},
    }
    summary = run_fall_risk_spearman_correlation(config)
    assert not summary.empty
    assert (metrics_dir / "fall_risk_spearman_correlation.csv").is_file()
    assert (metrics_dir / "fall_risk_spearman_correlation.md").is_file()
    assert (metrics_dir / "fall_risk_spearman_participant_means.csv").is_file()

    global_row = summary.loc[summary["comparison_scope"] == "global_all_participants"].iloc[0]
    assert global_row["defined"]
    assert float(global_row["spearman_rho"]) > 0.5

    per_cohort = summary[summary["comparison_scope"].str.startswith("healthy_vs_")]
    assert len(per_cohort) == 7

    payload = json.loads((metrics_dir / "fall_risk_spearman_correlation.json").read_text())
    assert "global_participant_spearman" in payload
