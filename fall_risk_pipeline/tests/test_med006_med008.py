"""Tests for MED-006 feature missingness and MED-008 SHAP sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.shap_sampling import stratified_shap_subject_order
from src.features.feature_missingness import (
    compute_feature_missingness_rows,
    warn_high_missingness_features,
    write_feature_missingness_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_compute_feature_missingness_flags_high_rate():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 1.0], [4.0, np.nan]])
    rows = compute_feature_missingness_rows(X, ["a", "b"], threshold=0.15)
    by_name = {r["feature"]: r for r in rows}
    assert by_name["b"]["exceeds_threshold"] is True
    assert by_name["a"]["exceeds_threshold"] is False


def test_warn_high_missingness_features():
    X = np.column_stack([np.ones(10), [np.nan] * 10])
    flagged = warn_high_missingness_features(
        X, ["ok", "bad"], threshold=0.15, context="test fold"
    )
    assert flagged == ["bad"]


def test_write_feature_missingness_report(tmp_path):
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 1.0]])
    df = write_feature_missingness_report(X, ["a", "b"], tmp_path)
    assert (tmp_path / "feature_missingness_report.csv").exists()
    assert len(df) == 2


def test_stratified_shap_subject_order_covers_all_cohorts():
    groups = np.array(
        [f"H{i}" for i in range(10)]
        + [f"P{i}" for i in range(10)]
        + [f"K{i}" for i in range(10)]
    )
    cohorts = np.array(["Healthy"] * 10 + ["PD"] * 10 + ["KneeOA"] * 10)
    order = stratified_shap_subject_order(groups, 15, cohorts=cohorts, seed=42)
    assert len(order) == 15
    cohorts_in = {cohorts[np.where(groups == s)[0][0]] for s in order}
    assert cohorts_in == {"Healthy", "PD", "KneeOA"}


def test_stratified_shap_all_subjects_when_cap_high():
    groups = np.array([f"p{i}" for i in range(20)])
    cohorts = np.array(["A"] * 10 + ["B"] * 10)
    order = stratified_shap_subject_order(groups, 260, cohorts=cohorts, seed=0)
    assert len(order) == 20


def test_pipeline_config_n_shap_samples_covers_n():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    assert int(cfg["explainability"]["n_shap_samples"]) >= 260
