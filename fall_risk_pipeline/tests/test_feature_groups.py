"""Tests for feature-group → patient-column mapping."""

from __future__ import annotations

from src.features.feature_groups import (
    count_trial_features,
    patient_columns_for_group,
    patient_columns_for_trial_bases,
    patient_columns_minus_group,
)


def _minimal_config() -> dict:
    return {
        "features": {
            "temporal": ["cadence", "stride_time_mean"],
            "spectral": ["spectral_centroid"],
            "trunk_dynamics": ["lyapunov", "apen"],
            "orientation": ["tilt_mean_deg"],
            "asymmetry": ["stride_time_asymmetry"],
            "turning": ["turn_duration_mean"],
            "spatial": [],
        }
    }


def test_count_trial_features():
    assert count_trial_features(_minimal_config()) == 8


def test_patient_columns_for_trial_base_aggregations():
    cols = [
        "cadence_mean",
        "cadence_std",
        "lyapunov_mean",
        "lyapunov_trend",
        "spectral_centroid_range",
        "unrelated_other",
    ]
    lyap = patient_columns_for_trial_bases(cols, ["lyapunov"])
    assert lyap == ["lyapunov_mean", "lyapunov_trend"]

    minus_spec = patient_columns_minus_group(cols, "spectral", _minimal_config())
    assert "spectral_centroid_range" not in minus_spec
    assert "cadence_mean" in minus_spec
