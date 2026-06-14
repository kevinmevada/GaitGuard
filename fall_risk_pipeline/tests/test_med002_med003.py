"""MED-002 / MED-003: binary strategy defaults and cohort AUC suppression threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.evaluator import Evaluator
from src.reporting.demographics_table import (
    build_demographics_by_cohort,
    demographics_to_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_pipeline_config_binary_strategy_threshold_ge_2():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    ds = cfg["dataset"]
    assert ds["binary_strategy"] == "threshold_ge_2"
    assert int(ds["high_risk_threshold"]) == 2


def test_pipeline_config_cohort_auc_min_n_at_least_25():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    assert int(cfg["models"]["evaluation"]["cohort_auc_min_n"]) >= 25


def test_demographics_table_includes_n_age_sex():
    trial_df = pd.DataFrame(
        {
            "participant_id": ["H1", "H2", "P1", "P2"],
            "cohort": ["Healthy", "Healthy", "PD", "PD"],
            "age": [65.0, 70.0, 72.0, 68.0],
            "sex": ["F", "M", "M", "F"],
            "laterality": [None, None, "Left", "Right"],
        }
    )
    table = build_demographics_by_cohort(trial_df)
    healthy = table[table["cohort"] == "Healthy"].iloc[0]
    assert int(healthy["n_participants"]) == 2
    assert "±" in str(healthy["age_mean_sd"])
    assert "F" in str(healthy["sex_ratio"])
    md = demographics_to_markdown(table)
    assert "MED-003" in md
    assert "n_participants" not in md  # rendered as markdown table columns


@pytest.fixture
def evaluator(tmp_path) -> Evaluator:
    config = {
        "paths": {
            "features": str(tmp_path / "features"),
            "checkpoints": str(tmp_path / "checkpoints"),
            "metrics": str(tmp_path / "metrics"),
            "figures_models": str(tmp_path / "figures_models"),
            "figures_shap": str(tmp_path / "figures_shap"),
        },
        "reporting": {"figure_format": "png", "figure_dpi": 100},
        "models": {
            "tuning": {"n_trials": 1, "timeout_per_model": 1, "cv_folds": 2},
            "evaluation": {
                "random_state": 42,
                "strategy": "nested_group_cv",
                "cohort_auc_min_n": 25,
            },
            "run": ["xgboost"],
            "ensemble": {"top_k": 1, "methods": []},
        },
        "dataset": {"label_mode": "binary"},
        "reproducibility": {"seed": 42},
    }
    return Evaluator(config)


def _binary_cohort_result(n: int, cohort: str = "HipOA") -> dict:
    y_true = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
    y_prob = np.linspace(0.05, 0.95, n)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "decision_threshold": 0.5,
        "cohorts": np.array([cohort] * n),
    }


def test_cohort_auc_suppressed_below_25(evaluator: Evaluator):
    rows = evaluator._cohort_metric_rows("xgboost", _binary_cohort_result(n=20))
    assert rows[0]["auc_status"] == "unstable_small_n"
    assert np.isnan(rows[0]["auc"])


def test_cohort_auc_stable_at_25(evaluator: Evaluator):
    rows = evaluator._cohort_metric_rows("xgboost", _binary_cohort_result(n=30))
    assert rows[0]["auc_status"] == "stable"
    assert np.isfinite(rows[0]["auc"])
