"""Tests for sensor-position ablation feature selection."""

from src.evaluation.sensor_ablation import (
    BILATERAL_ASYMMETRY,
    FOOT_BILATERAL,
    _features_for_sensors,
    _is_bilateral_foot_feature,
)


def test_bilateral_asymmetry_included_with_both_feet():
    feats = [
        "stride_time_mean_asymmetry_mean",
        "stride_time_std_asymmetry_std",
        "asymmetry_rms_acc_range",
        "left_stride_time_mean_mean",
        "head_rms_mean",
    ]
    idx = _features_for_sensors(
        feats, ("head", "lower_back", "left_foot", "right_foot")
    )
    selected = {feats[i] for i in idx}
    assert "stride_time_mean_asymmetry_mean" in selected
    assert "stride_time_std_asymmetry_std" in selected
    assert "asymmetry_rms_acc_range" in selected
    assert "left_stride_time_mean_mean" in selected


def test_bilateral_asymmetry_excluded_without_both_feet():
    feats = ["stride_time_mean_asymmetry_mean", "left_stride_time_mean_mean"]
    idx = _features_for_sensors(feats, ("left_foot",))
    selected = {feats[i] for i in idx}
    assert "stride_time_mean_asymmetry_mean" not in selected
    assert "left_stride_time_mean_mean" in selected


def test_bilateral_asymmetry_regex_matches_bare_trial_names():
    assert BILATERAL_ASYMMETRY.match("stride_time_mean_asymmetry")
    assert BILATERAL_ASYMMETRY.match("asymmetry_rms_acc_mean")
    assert FOOT_BILATERAL.match("cadence_mean_trend")
    assert _is_bilateral_foot_feature("stride_time_std_asymmetry_std")
