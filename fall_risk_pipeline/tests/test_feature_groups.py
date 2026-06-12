"""Tests for feature-group → patient-column mapping."""

from __future__ import annotations

from src.features.feature_groups import (
    count_trial_features,
    patient_columns_for_group,
    patient_columns_for_trial_bases,
    patient_columns_minus_group,
    patient_columns_minus_trial_bases,
)


def _minimal_config() -> dict:
    return {
        "features": {
            "temporal": ["cadence", "stride_time_mean"],
            "spectral": ["spectral_centroid"],
            "wavelet": ["wavelet_entropy"],
            "trunk_dynamics": [
                "lyapunov",
                "apen",
                "head_lb_rms_ratio",
                "head_lb_lyapunov_ratio",
                "head_lb_dfa_ratio",
            ],
            "orientation": ["tilt_mean_deg"],
            "asymmetry": ["stride_time_asymmetry"],
            "turning": ["turn_duration_mean"],
            "spatial": [],
        }
    }


def test_count_trial_features():
    assert count_trial_features(_minimal_config()) == 12


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


def test_minus_spectral_does_not_remove_wavelet_group():
    cols = [
        "lb_spectral_centroid_mean",
        "lb_wavelet_entropy_std",
        "cadence_mean",
    ]
    minus_spec = patient_columns_minus_group(cols, "spectral", _minimal_config())
    assert "lb_spectral_centroid_mean" not in minus_spec
    assert "lb_wavelet_entropy_std" in minus_spec

    minus_wav = patient_columns_minus_group(cols, "wavelet", _minimal_config())
    assert "lb_wavelet_entropy_std" not in minus_wav
    assert "lb_spectral_centroid_mean" in minus_wav


def test_head_lb_cross_site_requires_exact_base_match():
    cols = [
        "head_lb_rms_ratio_mean",
        "head_lb_lyapunov_ratio_std",
        "head_lb_dfa_ratio_trend",
        "lb_lyapunov_mean",
        "cadence_mean",
    ]
    trunk = patient_columns_for_group(cols, "trunk_dynamics", _minimal_config())
    assert "head_lb_rms_ratio_mean" in trunk
    assert "head_lb_lyapunov_ratio_std" in trunk
    assert "head_lb_dfa_ratio_trend" in trunk
    assert "lb_lyapunov_mean" in trunk

    minus_trunk = patient_columns_minus_group(cols, "trunk_dynamics", _minimal_config())
    assert "head_lb_rms_ratio_mean" not in minus_trunk
    assert "head_lb_lyapunov_ratio_std" not in minus_trunk
    assert "cadence_mean" in minus_trunk

    minus_lyap = patient_columns_minus_trial_bases(cols, ["lyapunov"])
    assert "lb_lyapunov_mean" not in minus_lyap
    assert "head_lb_lyapunov_ratio_std" in minus_lyap
    assert "head_lb_rms_ratio_mean" in minus_lyap
