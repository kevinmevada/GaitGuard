"""Spectral feature helpers (centroid vs dominant frequency)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.features.feature_extractor import FeatureExtractor, spectral_centroid_hz


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
