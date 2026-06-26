"""Stage C unified accelerometer bandpass (0.5–20 Hz, filtfilt)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.unified_bandpass import (
    UnifiedBandpassConfig,
    apply_unified_acc_bandpass,
    bandpass_coefficients,
    filtfilt_bandpass,
)
from scipy.signal import lfilter


def _sine(fs: float, freq: float, duration_s: float = 10.0) -> np.ndarray:
    t = np.arange(0, duration_s, 1.0 / fs)
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_coefficients_match_spec():
    order = 4
    b, a = bandpass_coefficients(fs_hz=100.0, low_hz=0.5, high_hz=20.0, order=order)
    assert len(b) == 2 * order + 1
    assert len(a) == 2 * order + 1


def test_filtfilt_preserves_gait_band_more_than_dc():
    fs = 100.0
    sig = _sine(fs, 0.1) + _sine(fs, 5.0)
    out = filtfilt_bandpass(sig, fs_hz=fs, low_hz=0.5, high_hz=20.0, order=4)
    # 5 Hz component should dominate after bandpass
    t = np.arange(len(sig)) / fs
    fit_5hz = np.sin(2 * np.pi * 5.0 * t)
    corr_pass = np.corrcoef(out[len(out) // 4 :], fit_5hz[len(out) // 4 :])[0, 1]
    assert corr_pass > 0.85


def test_filtfilt_zero_phase_vs_lfilter_group_delay():
    fs = 100.0
    sig = _sine(fs, 5.0, duration_s=4.0)
    out_filtfilt = filtfilt_bandpass(sig, fs_hz=fs)
    b, a = bandpass_coefficients(fs_hz=fs)
    out_lfilter = lfilter(b, a, sig)
    # Zero-phase: peak correlation at zero lag
    cc_ff = np.correlate(out_filtfilt - out_filtfilt.mean(), sig - sig.mean(), mode="full")
    lag_ff = int(np.argmax(cc_ff)) - (len(sig) - 1)
    cc_lf = np.correlate(out_lfilter - out_lfilter.mean(), sig - sig.mean(), mode="full")
    lag_lf = int(np.argmax(cc_lf)) - (len(sig) - 1)
    assert abs(lag_ff) < 3
    assert abs(lag_lf) > abs(lag_ff)


def test_apply_unified_identical_voisard_and_daphnet_frames():
    fs = 100.0
    cfg = UnifiedBandpassConfig(fs_hz=fs)
    n = 500
    t = np.arange(n) / fs
    base = pd.DataFrame(
        {
            "time": t,
            "acc_x": np.sin(2 * np.pi * 2 * t),
            "acc_y": np.sin(2 * np.pi * 2.2 * t),
            "acc_z": np.sin(2 * np.pi * 1.8 * t),
        }
    )
    voisard = base.copy()
    voisard["gyr_x"] = 0.0
    daphnet = base.copy()

    out_v = apply_unified_acc_bandpass(voisard, cfg)
    out_d = apply_unified_acc_bandpass(daphnet, cfg)
    for col in ("acc_x", "acc_y", "acc_z"):
        assert np.allclose(out_v[col].to_numpy(), out_d[col].to_numpy())


def test_signal_processor_uses_unified_bandpass():
    from pathlib import Path

    from src.preprocessing.signal_processor import SignalProcessor

    source = (Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "signal_processor.py").read_text(
        encoding="utf-8"
    )
    assert "apply_unified_acc_bandpass" in source
    assert "Stage C" in source

    sp = SignalProcessor(
        {
            "paths": {"processed_data": "."},
            "dataset": {"sampling_rate": 100},
            "preprocessing": {
                "unified_acc_bandpass": {
                    "enabled": True,
                    "low_hz": 0.5,
                    "high_hz": 20.0,
                    "order": 4,
                },
                "madgwick_enabled": False,
                "madgwick_beta": 0.1,
                "gait_event_source": "algorithm",
                "max_nan_fraction_before_filter": 0.05,
            },
        }
    )
    assert sp.bandpass_cfg.low_hz == 0.5
    assert sp.bandpass_cfg.high_hz == 20.0
