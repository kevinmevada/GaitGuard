"""Spectral feature helpers (centroid vs dominant frequency)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from src.features.feature_extractor import (
    FeatureExtractor,
    harmonic_ratio_even_odd,
    spectral_centroid_hz,
)


def test_spectral_centroid_pure_tone_near_peak_frequency():
    fs = 100.0
    f0 = 2.5
    t = np.arange(0, 8, 1 / fs)
    sig = np.sin(2 * np.pi * f0 * t)
    freqs, pxx = welch(sig, fs=fs, nperseg=256)
    centroid = spectral_centroid_hz(freqs, pxx)
    assert np.isfinite(centroid)
    assert abs(centroid - f0) < 0.35


def test_spectral_features_include_centroid():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    sig = np.sin(2 * np.pi * 1.5 * t)
    df = pd.DataFrame({"acc_z_grav_free": sig, "time": t})
    cfg = {"paths": {"processed_data": ".", "features": "."}, "dataset": {"sampling_rate": fs}, "features": {}}
    ext = FeatureExtractor(cfg)
    feats = ext._spectral_features(df, prefix="lb")
    assert "lb_spectral_centroid" in feats
    assert "lb_dominant_freq" in feats
    assert "lb_spectral_entropy" in feats
    assert np.isfinite(feats["lb_spectral_centroid"])


def test_harmonic_ratio_excludes_harmonics_above_lowpass_band():
    fs = 100.0
    lp_cut = 15.0
    nperseg = 256
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    pxx = np.ones_like(freqs)

    # 1.8 Hz cadence: 6th harmonic = 10.8 Hz (in band); unrestricted sum uses k=1..6.
    dominant = 1.8
    unrestricted = [
        float(pxx[np.argmin(np.abs(freqs - dominant * k))]) for k in range(1, 7)
    ]
    unrestricted_ratio = sum(unrestricted[1::2]) / (sum(unrestricted[0::2]) + 1e-10)

    filtered = harmonic_ratio_even_odd(freqs, pxx, dominant, lp_cut)
    assert filtered is not None
    # 7th harmonic not computed; 6th at 10.8 Hz stays (< 12 Hz cutoff).
    assert filtered == pytest.approx(unrestricted_ratio)

    # 2.5 Hz cadence: k=5,6 fall above 12 Hz; with rising PSD they change the ratio.
    dominant_fast = 2.5
    included = [k for k in range(1, 7) if dominant_fast * k < lp_cut * 0.8]
    pxx_weighted = np.linspace(1.0, 10.0, len(freqs))
    filtered_fast = harmonic_ratio_even_odd(freqs, pxx_weighted, dominant_fast, lp_cut)
    unrestricted_fast = (
        sum(float(pxx_weighted[np.argmin(np.abs(freqs - dominant_fast * k))]) for k in range(2, 7, 2))
        / (
            sum(float(pxx_weighted[np.argmin(np.abs(freqs - dominant_fast * k))]) for k in range(1, 7, 2))
            + 1e-10
        )
    )
    assert filtered_fast is not None
    assert included == [1, 2, 3, 4]
    assert filtered_fast != pytest.approx(unrestricted_fast)
