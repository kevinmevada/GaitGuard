"""Tests for compute overhead (training & inference latency) reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_time_callable_median():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.compute_timing import median_seconds, time_callable

    counter = {"n": 0}

    def fn() -> int:
        counter["n"] += 1
        return counter["n"]

    result, med, all_t = time_callable(fn, n_warmup=1, n_repeat=3)
    assert result == 4
    assert len(all_t) == 3
    assert med == median_seconds(all_t)


def test_run_compute_overhead_classical_only(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.compute_overhead_evaluator import run_compute_overhead_benchmark

    rng = np.random.default_rng(0)
    n = 16
    X = rng.normal(size=(n, 5))
    y = np.array([0] * 8 + [1] * 8)
    groups = np.array([f"p{i // 4}" for i in range(n)])
    meta = pd.DataFrame(
        {
            "trial_id": [f"t{i}" for i in range(n)],
            "participant_id": groups,
            "cohort": ["Healthy"] * 8 + ["PD"] * 8,
            "risk_label": y,
        }
    )

    metrics_dir = tmp_path / "metrics"
    config = {
        "paths": {"metrics": str(metrics_dir), "processed_data": str(tmp_path / "proc")},
        "reproducibility": {"seed": 42},
        "classical_baselines": {
            "enabled": True,
            "models": ["logistic_regression_l2"],
            "export_oof_scores": False,
        },
        "dl_baselines": {"enabled": False},
        "compute_overhead": {
            "enabled": True,
            "classical": True,
            "dl_rocket": False,
            "bilstm_ae": False,
            "n_warmup": 0,
            "n_inference_repeats": 2,
        },
    }

    with patch(
        "src.evaluation.compute_overhead_evaluator._load_matrix",
        return_value=(X, y, groups, ["f0", "f1", "f2", "f3", "f4"], meta),
    ):
        df = run_compute_overhead_benchmark(config)

    assert not df.empty
    assert (metrics_dir / "compute_overhead_metrics.csv").is_file()
    assert (metrics_dir / "compute_overhead_summary.json").is_file()
    assert (metrics_dir / "compute_overhead_report.md").is_file()
    row = df.iloc[0]
    assert row["train_time_s_per_fold"] > 0
    assert row["inference_ms_per_unit"] > 0
    assert row["inference_unit"] == "trial"

    summary = json.loads((metrics_dir / "compute_overhead_summary.json").read_text())
    assert "device" in summary["host"]
