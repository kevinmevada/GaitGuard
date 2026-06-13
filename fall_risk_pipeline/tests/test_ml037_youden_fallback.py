"""ML-037: no in-sample Youden when inner grouped OOF is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.clinical_threshold import (
    DEFAULT_DECISION_THRESHOLD,
    THRESHOLD_STRATEGY_FIXED_INSUFFICIENT_GROUPS,
    THRESHOLD_STRATEGY_FIXED_OOF_UNAVAILABLE,
    THRESHOLD_STRATEGY_INNER_GROUP_OOF,
    fixed_threshold_when_inner_cv_unavailable,
    threshold_from_inner_oof,
    youden_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fixed_threshold_when_inner_cv_unavailable():
    thresh, strategy = fixed_threshold_when_inner_cv_unavailable()
    assert thresh == DEFAULT_DECISION_THRESHOLD
    assert strategy == THRESHOLD_STRATEGY_FIXED_INSUFFICIENT_GROUPS


def test_small_groups_would_have_optimistic_in_sample_youden():
    """Prove fixed 0.5 differs from in-sample Youden on separable train scores."""
    y = np.array([0, 0, 1, 1])
    in_sample = np.array([0.05, 0.15, 0.85, 0.95])
    assert youden_threshold(y, in_sample) != DEFAULT_DECISION_THRESHOLD
    assert fixed_threshold_when_inner_cv_unavailable()[0] == DEFAULT_DECISION_THRESHOLD


def test_threshold_from_inner_oof_uses_youden_when_valid():
    y = np.repeat([0, 1], 20)
    oof = np.where(y == 0, 0.2, 0.8).astype(float)
    thresh, strategy = threshold_from_inner_oof(y, oof, n_splits=5)
    assert strategy == THRESHOLD_STRATEGY_INNER_GROUP_OOF
    assert 0.0 < thresh < 1.0


def test_threshold_from_inner_oof_sparse_returns_fixed_flag():
    y = np.array([0, 0, 1, 1])
    oof = np.array([np.nan, np.nan, 0.9, 0.8])
    thresh, strategy = threshold_from_inner_oof(y, oof, n_splits=5)
    assert thresh == DEFAULT_DECISION_THRESHOLD
    assert strategy == THRESHOLD_STRATEGY_FIXED_OOF_UNAVAILABLE


def test_evaluator_and_ablation_drop_in_sample_train_fallback():
    evaluator = (REPO_ROOT / "fall_risk_pipeline" / "src" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8"
    )
    ablation = (
        REPO_ROOT / "fall_risk_pipeline" / "src" / "evaluation" / "feature_ablation.py"
    ).read_text(encoding="utf-8")
    assert "in_sample_train_fallback" not in evaluator
    assert "in_sample_soft_voting_fallback" not in evaluator
    assert "fixed_threshold_when_inner_cv_unavailable" in evaluator
    assert "threshold_from_inner_oof" in evaluator
    assert "fixed_threshold_when_inner_cv_unavailable" in ablation
    assert "predict_proba(X_tr)" not in ablation.split("_train_fold_youden_binary")[1].split("def ")[0]
