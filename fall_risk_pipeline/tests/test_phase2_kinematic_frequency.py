"""Phase 2 kinematic, kinetic & frequency-domain features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from src.features.phase2_kinematic_frequency import (
    Phase2Config,
    extract_phase2_kinematic_frequency_features,
)
from src.features.spectral_utils import (
    freezing_index_from_psd,
    sample_freezing_index_series,
)


def test_freezing_index_higher_with_high_frequency_content():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    loco = np.sin(2 * np.pi * 1.0 * t)
    freeze = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 5.0 * t)
    freqs_l, pxx_l = welch(loco, fs=fs, nperseg=256)
    freqs_f, pxx_f = welch(freeze, fs=fs, nperseg=256)
    fi_loco = freezing_index_from_psd(freqs_l, pxx_l)
    fi_freeze = freezing_index_from_psd(freqs_f, pxx_f)
    assert np.isfinite(fi_loco) and np.isfinite(fi_freeze)
    assert fi_freeze > fi_loco


def test_sample_freezing_index_series_length():
    fs = 100.0
    sig = np.random.default_rng(0).normal(size=500)
    fi = sample_freezing_index_series(sig, fs, window_s=2.0)
    assert len(fi) == len(sig)
    assert np.isfinite(fi).any()


def test_peak_gyro_and_rms_window_features():
    fs = 100.0
    n = 400
    gyr_rad = np.ones(n) * np.deg2rad(90.0)
    acc = np.sin(2 * np.pi * 2 * np.arange(n) / fs)
    df = pd.DataFrame(
        {
            "gyr_x": gyr_rad,
            "gyr_y": gyr_rad * 0.5,
            "gyr_z": gyr_rad * 0.2,
            "gyr_resultant": np.linalg.norm(
                np.column_stack([gyr_rad, gyr_rad * 0.5, gyr_rad * 0.2]), axis=1
            ),
            "acc_resultant": acc,
        }
    )
    signals = {"lower_back": df}
    feats = extract_phase2_kinematic_frequency_features(
        signals, fs=fs, config=Phase2Config(gyro_in_degrees=False)
    )
    assert feats["lb_peak_gyro_deg_s"] == pytest.approx(102.22, rel=0.02)
    assert feats["lb_rms_acc_2s_mean"] > 0.0


def test_lb_harmonic_ratios_and_freezing_index():
    fs = 100.0
    t = np.arange(0, 8, 1 / fs)
    sig = np.sin(2 * np.pi * 1.2 * t)
    df = pd.DataFrame(
        {
            "acc_x_grav_free": sig,
            "acc_y_grav_free": sig * 0.8,
            "acc_z_grav_free": sig * 0.6,
        }
    )
    feats = extract_phase2_kinematic_frequency_features(
        {"lower_back": df}, fs=fs, config=Phase2Config()
    )
    assert "lb_harmonic_ratio_ap" in feats
    assert "lb_harmonic_ratio_ml" in feats
    assert "lb_harmonic_ratio_v" in feats
    assert "lb_freezing_index" in feats


def test_joint_angles_from_swing_gyro():
    fs = 100.0
    n = 600
    hs = list(range(0, n, 100))
    to = [h + 40 for h in hs[:-1]]

    def foot_df(side: str) -> pd.DataFrame:
        gyr = np.zeros(n)
        for h, t0 in zip(hs[:-1], to):
            swing_len = h + 100 - t0
            gyr[t0 : h + 100] = np.deg2rad(60.0) * np.sin(
                np.linspace(0, np.pi, swing_len)
            )
        df = pd.DataFrame({f"heel_strike_{side}": 0, f"toe_off_{side}": 0, "gyr_y": gyr})
        df.loc[hs, f"heel_strike_{side}"] = 1
        df.loc[to, f"toe_off_{side}"] = 1
        return df

    lb = pd.DataFrame(
        {"gyr_y": np.deg2rad(10.0) * np.sin(2 * np.pi * 1.0 * np.arange(n) / fs)}
    )
    signals = {"lower_back": lb, "left_foot": foot_df("left"), "right_foot": foot_df("right")}
    feats = extract_phase2_kinematic_frequency_features(
        signals, fs=fs, config=Phase2Config(gyro_in_degrees=False)
    )
    assert feats["ankle_angle_rom_deg"] > 0.5
    assert feats["knee_angle_rom_deg"] > 0.0
