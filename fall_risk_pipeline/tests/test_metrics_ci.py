"""Tests for src/evaluation/metrics_ci.py.

This module previously had no dedicated test file, even though it backs
every bootstrap-CI reported in the pipeline's metrics tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.metrics_ci import (  # noqa: E402
    grouped_bootstrap_binary_auc_ci,
    subject_bootstrap_binary_auc_ci,
)


def _separable_binary_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = rng.random(n)
    score[y == 1] += 0.4
    return y, np.clip(score, 0.01, 0.99)


def test_subject_bootstrap_returns_four_tuple_with_point_estimate():
    y, score = _separable_binary_data()
    auc = roc_auc_score(y, score)
    auc_full, low, high, status = subject_bootstrap_binary_auc_ci(y, score, seed=42)
    assert status == "subject_bootstrap"
    assert auc_full == auc
    assert low <= auc <= high


def test_subject_bootstrap_insufficient_data():
    y = np.array([0, 0])
    score = np.array([0.1, 0.9])
    auc_full, low, high, status = subject_bootstrap_binary_auc_ci(y, score, seed=1)
    assert status == "insufficient_data"
    assert np.isnan(auc_full) and np.isnan(low) and np.isnan(high)


def test_subject_bootstrap_reproducible_with_same_seed():
    y, score = _separable_binary_data()
    a = subject_bootstrap_binary_auc_ci(y, score, seed=7)
    b = subject_bootstrap_binary_auc_ci(y, score, seed=7)
    assert a == b


def test_grouped_bootstrap_matches_point_estimate():
    y, score = _separable_binary_data(n=120)
    # 20 subjects, 6 windows each
    groups = np.repeat(np.arange(20), 6)
    auc = roc_auc_score(y, score)
    auc_full, low, high, status = grouped_bootstrap_binary_auc_ci(
        y, score, groups, seed=42
    )
    assert status == "subject_cluster_bootstrap"
    assert auc_full == auc
    assert low <= auc <= high
    assert np.isfinite(low) and np.isfinite(high)


def test_grouped_bootstrap_insufficient_groups():
    y, score = _separable_binary_data(n=10)
    groups = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])  # only 2 unique groups
    auc_full, low, high, status = grouped_bootstrap_binary_auc_ci(
        y, score, groups, seed=1
    )
    assert status == "insufficient_groups"
    assert np.isfinite(auc_full)
    assert np.isnan(low) and np.isnan(high)


def test_grouped_bootstrap_reproducible_with_same_seed():
    y, score = _separable_binary_data(n=120)
    groups = np.repeat(np.arange(20), 6)
    a = grouped_bootstrap_binary_auc_ci(y, score, groups, seed=3)
    b = grouped_bootstrap_binary_auc_ci(y, score, groups, seed=3)
    assert a == b
