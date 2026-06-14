"""ML-044: suppress numeric cohort AUC when auc_status is unstable_small_n."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = REPO_ROOT / "fall_risk_pipeline" / "src" / "evaluation" / "evaluator.py"


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


def test_unstable_small_n_cohort_masks_auc(evaluator: Evaluator):
    rows = evaluator._cohort_metric_rows("xgboost", _binary_cohort_result(n=10))
    assert len(rows) == 1
    row = rows[0]
    assert row["auc_status"] == "unstable_small_n"
    assert row["n"] == 10
    assert np.isnan(row["auc"])
    assert np.isnan(row["auc_ci_low"])
    assert np.isnan(row["auc_ci_high"])
    assert np.isnan(row["auc_pr"])
    assert np.isfinite(row["f1"])
    assert np.isfinite(row["accuracy"])


def test_stable_cohort_keeps_numeric_auc(evaluator: Evaluator):
    rows = evaluator._cohort_metric_rows("xgboost", _binary_cohort_result(n=30))
    assert len(rows) == 1
    row = rows[0]
    assert row["auc_status"] == "stable"
    assert np.isfinite(row["auc"])
    assert np.isfinite(row["auc_ci_low"])
    assert np.isfinite(row["auc_ci_high"])


def test_cohort_metric_rows_source_masks_unstable_auc():
    tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_cohort_metric_rows":
            source = ast.get_source_segment(
                EVALUATOR_PATH.read_text(encoding="utf-8"), node
            ) or ""
            assert "unstable_small_n" in source
            assert "float(\"nan\")" in source or "float('nan')" in source
            return
    raise AssertionError("_cohort_metric_rows not found")
