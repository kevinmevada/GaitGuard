"""Tests for patient-level range and trend aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.patient_temporal_aggregation import (
    aggregate_trial_values,
    order_trial_group,
    trial_feature_range,
    trial_feature_trend_slope,
)


def test_range_single_trial_is_zero():
    assert trial_feature_range(np.array([3.5])) == 0.0


def test_range_multiple_trials():
    assert trial_feature_range(np.array([1.0, 4.0, 2.0])) == pytest.approx(3.0)


def test_trend_slope_increasing():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert trial_feature_trend_slope(y) == pytest.approx(1.0, rel=1e-5)


def test_trend_requires_min_trials():
    assert np.isnan(trial_feature_trend_slope(np.array([5.0]), min_trials=2))


def test_order_trial_group_by_session():
    df = pd.DataFrame({
        "trial_id": ["HS_2_3", "HS_2_1", "HS_2_2"],
        "session": [1, 1, 1],
        "x": [30, 10, 20],
    })
    ordered = order_trial_group(df, ["session", "trial_id"])
    assert list(ordered["x"]) == [10, 20, 30]


def test_aggregate_trial_values_all_stats():
    cfg = {
        "include_mean": True,
        "include_std": True,
        "include_range": True,
        "include_trend": True,
        "min_trials_for_trend": 2,
    }
    stats = aggregate_trial_values(np.array([1.0, 2.0, 3.0, 4.0]), cfg)
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["range"] == pytest.approx(3.0)
    assert stats["trend"] == pytest.approx(1.0, rel=1e-5)
