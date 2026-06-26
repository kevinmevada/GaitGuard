"""Tests for Wilcoxon/Holm + Critical Difference statistical benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _write_synthetic_oof(
    metrics_dir: Path,
    model: str,
    *,
    seed: int,
    bias: float = 0.0,
    n_participants: int = 12,
    trials_per_participant: int = 3,
) -> None:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for p in range(n_participants):
        pid = f"p{p}"
        label = 0 if p < n_participants // 2 else 1
        for t in range(trials_per_participant):
            score = float(rng.normal(label * 0.6 + bias, 0.25))
            rows.append(
                {
                    "trial_id": f"{pid}_t{t}",
                    "participant_id": pid,
                    "y_true": label,
                    "score": score,
                }
            )
    oof_dir = metrics_dir / "oof_scores"
    oof_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(oof_dir / f"{model}.csv", index=False)


def test_leave_one_participant_out_aurocs_paired():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.loso_oof_scores import align_jackknife_aurocs, leave_one_participant_out_aurocs

    rng = np.random.default_rng(0)
    n = 40
    y = np.array([0] * 20 + [1] * 20)
    pids = np.array([f"p{i // 4}" for i in range(n)])
    s_a = rng.normal(y * 0.5, 0.2)
    s_b = rng.normal(y * 0.4, 0.2)

    auc_a, pa = leave_one_participant_out_aurocs(y, s_a, pids)
    auc_b, pb = leave_one_participant_out_aurocs(y, s_b, pids)
    common, aligned = align_jackknife_aurocs(
        {"a": (auc_a, pa), "b": (auc_b, pb)},
    )
    assert len(common) >= 3
    assert len(aligned["a"]) == len(aligned["b"]) == len(common)


def test_holm_wilcoxon_vs_reference():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.critical_difference import holm_correction, wilcoxon_vs_reference

    rng = np.random.default_rng(1)
    n = 10
    ref = rng.uniform(0.7, 0.95, n)
    base1 = ref - 0.02 + rng.normal(0, 0.01, n)
    base2 = ref - 0.15 + rng.normal(0, 0.01, n)
    aligned = {"bilstm_ae_ensemble": ref, "svm_rbf": base1, "rocket": base2}
    df = wilcoxon_vs_reference(aligned, "bilstm_ae_ensemble", alpha=0.05)
    assert len(df) == 2
    assert "p_holm" in df.columns
    assert "significant_holm" in df.columns

    p_adj, reject = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
    assert len(p_adj) == 3


def test_nemenyi_cd_and_friedman():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.critical_difference import build_cd_summary, nemenyi_critical_difference

    k, n = 4, 8
    scores = np.random.default_rng(2).uniform(0.6, 0.95, size=(k, n))
    names = ["m1", "m2", "m3", "m4"]
    summary = build_cd_summary({n: scores[i] for i, n in enumerate(names)}, names, alpha=0.05)
    assert summary["critical_difference"] > 0
    assert summary["friedman"]["n_datasets"] == n
    cd = nemenyi_critical_difference(4, 10)
    assert cd > 0


def test_run_statistical_benchmark_exports_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.statistical_benchmark_evaluator import run_statistical_benchmark

    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    _write_synthetic_oof(metrics_dir, "bilstm_ae_ensemble", seed=0, bias=0.05)
    _write_synthetic_oof(metrics_dir, "svm_rbf", seed=1, bias=-0.05)
    _write_synthetic_oof(metrics_dir, "rocket", seed=2, bias=-0.12)

    config = {
        "paths": {"metrics": str(metrics_dir), "figures_models": str(figures_dir)},
        "statistical_benchmark": {
            "enabled": True,
            "reference_model": "bilstm_ae_ensemble",
            "alpha": 0.05,
            "models": ["bilstm_ae_ensemble", "svm_rbf", "rocket"],
        },
    }
    summary = run_statistical_benchmark(config)
    assert summary
    assert (metrics_dir / "wilcoxon_vs_bilstm_ae.csv").is_file()
    assert (metrics_dir / "jackknife_auroc_by_fold.csv").is_file()
    assert (metrics_dir / "statistical_benchmark_summary.json").is_file()
    assert (metrics_dir / "statistical_benchmark_report.md").is_file()
    assert (figures_dir / "critical_difference_auroc.png").is_file()

    wilcoxon = pd.read_csv(metrics_dir / "wilcoxon_vs_bilstm_ae.csv")
    assert "p_holm" in wilcoxon.columns
