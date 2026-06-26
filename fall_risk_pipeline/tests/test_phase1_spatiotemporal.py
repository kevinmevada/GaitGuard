"""Phase 1 spatiotemporal + variability features (Moon / Trabassi alignment)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.phase1_spatiotemporal import (
    Phase1Config,
    coefficient_of_variation_pct,
    extract_phase1_spatiotemporal_features,
    rolling_cv_pct,
    symmetry_index,
)


def test_symmetry_index_identical_sides_is_zero():
    assert symmetry_index(1.0, 1.0) == pytest.approx(0.0)


def test_symmetry_index_formula():
    # |80 − 120| / (0.5 × 200) × 100 = 40%
    assert symmetry_index(80.0, 120.0) == pytest.approx(40.0)


def test_coefficient_of_variation_pct():
    vals = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
    cv = coefficient_of_variation_pct(vals)
    assert cv == pytest.approx(np.std(vals) / np.mean(vals) * 100.0)


def test_rolling_cv_pct_uses_window():
    vals = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.05])
    full = coefficient_of_variation_pct(vals)
    rolling = rolling_cv_pct(vals, window=3)
    assert np.isfinite(rolling)
    assert rolling != pytest.approx(full) or len(vals) == 3


def _synthetic_foot(
    *,
    fs: float,
    n_strides: int,
    stride_s: float,
    stance_frac: float,
    side: str,
    start_offset: int = 0,
) -> pd.DataFrame:
    stride_samples = int(stride_s * fs)
    stance_samples = int(stance_frac * stride_samples)
    n = start_offset + n_strides * stride_samples + stride_samples
    acc = np.zeros(n, dtype=float)
    hs_col = f"heel_strike_{side}"
    to_col = f"toe_off_{side}"

    hs_idx = [start_offset + i * stride_samples for i in range(n_strides)]
    to_idx = [h + stance_samples for h in hs_idx]

    for h, t in zip(hs_idx, to_idx):
        end = min(h + stride_samples, n)
        swing_len = end - t
        if swing_len > 4:
            acc[t:end] = 3.0 * np.sin(np.linspace(0.0, np.pi, swing_len))

    df = pd.DataFrame(
        {
            "acc_resultant": acc,
            hs_col: 0,
            to_col: 0,
        }
    )
    df.loc[hs_idx, hs_col] = 1
    df.loc[to_idx, to_col] = 1
    return df


def test_extract_phase1_timing_features():
    fs = 100.0
    stride_s = 1.0
    lf = _synthetic_foot(
        fs=fs, n_strides=6, stride_s=stride_s, stance_frac=0.6, side="left", start_offset=0
    )
    rf = _synthetic_foot(
        fs=fs,
        n_strides=6,
        stride_s=stride_s,
        stance_frac=0.6,
        side="right",
        start_offset=int(0.5 * stride_s * fs),
    )
    cfg = {"features": {"phase1_spatiotemporal": {"min_strides": 3}}}
    feats = extract_phase1_spatiotemporal_features(lf, rf, fs=fs, config=cfg)

    assert feats["left_stride_duration_s"] == pytest.approx(stride_s, rel=0.02)
    assert feats["right_stride_duration_s"] == pytest.approx(stride_s, rel=0.02)
    assert feats["step_duration_s"] == pytest.approx(0.5 * stride_s, rel=0.05)
    assert 50 < feats["left_stance_pct"] < 70
    assert 30 < feats["left_swing_pct"] < 50
    assert feats["stride_duration_cv_pct"] >= 0.0
    assert feats["si_stride_duration"] == pytest.approx(0.0, abs=1.0)
    assert "stride_time_mean" in feats
    assert feats["stride_time_mean"] == pytest.approx(stride_s, rel=0.02)


def test_extract_phase1_spatial_integration_enabled():
    fs = 100.0
    lf = _synthetic_foot(fs=fs, n_strides=8, stride_s=1.0, stance_frac=0.6, side="left")
    rf = _synthetic_foot(
        fs=fs,
        n_strides=8,
        stride_s=1.0,
        stance_frac=0.6,
        side="right",
        start_offset=50,
    )
    feats = extract_phase1_spatiotemporal_features(
        lf,
        rf,
        fs=fs,
        config=Phase1Config(enable_spatial_integration=True, min_strides=3),
    )
    assert feats["left_step_length_m"] > 1e-4
    assert feats["right_step_length_m"] > 1e-4
    assert feats["gait_velocity_m_s"] > 1e-4
    assert np.isfinite(feats["si_step_length"])


def test_phase1_disabled_returns_empty():
    lf = _synthetic_foot(fs=100, n_strides=4, stride_s=1.0, stance_frac=0.6, side="left")
    rf = _synthetic_foot(
        fs=100, n_strides=4, stride_s=1.0, stance_frac=0.6, side="right", start_offset=50
    )
    feats = extract_phase1_spatiotemporal_features(
        lf, rf, fs=100.0, config=Phase1Config(enabled=False)
    )
    assert feats == {}
