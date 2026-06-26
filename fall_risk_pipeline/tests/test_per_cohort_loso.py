"""Tests for per-cohort LOSO pathology-tier reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _synthetic_oof(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    spec = [
        ("Healthy", 0, 8, 0.15),
        ("PD", 1, 6, 0.25),
        ("CVA", 1, 6, 0.55),
        ("HipOA", 1, 6, 0.50),
        ("KneeOA", 1, 6, 0.48),
        ("ACL", 1, 6, 0.45),
        ("CIPN", 1, 4, 0.60),
        ("RIL", 1, 4, 0.58),
    ]
    tid = 0
    for cohort, label, n_trials, base in spec:
        for p in range(2):
            pid = f"{cohort[:2]}_{p}"
            for _ in range(n_trials // 2):
                score = float(base + rng.normal(0, 0.08) + label * 0.1)
                rows.append(
                    {
                        "trial_id": f"t{tid}",
                        "participant_id": pid,
                        "cohort": cohort,
                        "eval_non_healthy": label,
                        "bilstm_ae_ensemble_score": score,
                    }
                )
                tid += 1

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    path = metrics_dir / "bilstm_ae_loso_oof_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return metrics_dir


def test_kruskal_wallis_and_one_vs_healthy():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.per_cohort_loso_metrics import (
        kruskal_wallis_across_cohorts,
        one_vs_healthy_metrics,
    )

    cohorts = np.array(["Healthy", "Healthy", "PD", "PD", "CVA", "CVA"])
    scores = np.array([0.1, 0.12, 0.2, 0.22, 0.7, 0.75])
    row = one_vs_healthy_metrics(cohorts, scores, "CVA")
    assert row["auroc"] > 0.9
    assert row["f1_binary"] > 0.5
    kw = kruskal_wallis_across_cohorts(cohorts, scores)
    assert kw["n_groups"] >= 2


def test_pd_paradox_note():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.per_cohort_loso_metrics import pd_clinical_paradox_note

    pd_row = {
        "cohort": "PD",
        "anomaly_rate_pct": 15.0,
        "reference_fall_probability_pct": 67.3,
    }
    others = [
        {"cohort": "CVA", "anomaly_rate_pct": 55.0},
        {"cohort": "HipOA", "anomaly_rate_pct": 50.0},
    ]
    note = pd_clinical_paradox_note(pd_row, others)
    assert note is not None
    assert "PD clinical paradox" in note


def test_run_per_cohort_loso_exports_detailed_report(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.per_cohort_loso_evaluator import run_per_cohort_loso_results

    metrics_dir = _synthetic_oof(tmp_path)
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "_pipeline_meta": {
            "config_path": str(PIPELINE_ROOT / "configs" / "pipeline_config.yaml"),
        },
        "per_cohort_loso": {"enabled": True, "sync_paper_docs": True},
    }
    df = run_per_cohort_loso_results(config)

    assert not df.empty
    assert len(df) == 7
    assert (metrics_dir / "per_cohort_loso_metrics.csv").is_file()
    assert (metrics_dir / "per_cohort_loso_results.md").is_file()
    assert (metrics_dir / "per_cohort_kruskal_wallis.json").is_file()

    md = (metrics_dir / "per_cohort_loso_results.md").read_text(encoding="utf-8")
    assert "Kruskal-Wallis" in md
    assert "HOA" in md or "HipOA" in md
    assert "one-vs-Healthy" in md.lower() or "vs Healthy" in md

    kw = json.loads((metrics_dir / "per_cohort_kruskal_wallis.json").read_text())
    assert "trial_level" in kw

    paper = REPO_ROOT / "docs" / "paper" / "per_cohort_loso_results.md"
    assert paper.is_file()
