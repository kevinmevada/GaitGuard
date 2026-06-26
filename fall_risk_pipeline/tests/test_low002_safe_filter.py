"""LOW-002: _safe_filter NaN handling without flat bfill/ffill."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.signal_processor import SignalProcessor


def _processor() -> SignalProcessor:
    return SignalProcessor(
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
                "lowpass_cutoff_hz": 20.0,
                "highpass_cutoff_hz": 0.5,
                "lowpass_order": 4,
                "madgwick_beta": 0.1,
                "madgwick_enabled": False,
                "gait_event_source": "algorithm",
                "max_nan_fraction_before_filter": 0.05,
            },
        }
    )


def test_safe_filter_discards_high_nan_fraction():
    sp = _processor()
    n = 100
    df = pd.DataFrame({
        "acc_x": [np.nan] * 6 + list(np.linspace(0, 1, n - 6)),
        "acc_y": [np.nan] * 6 + list(np.zeros(n - 6)),
        "acc_z": [np.nan] * 6 + list(np.zeros(n - 6)),
        "gyr_x": [np.nan] * 6 + list(np.zeros(n - 6)),
    })
    out = sp._safe_filter(df)
    assert out.empty


def test_safe_filter_interpolates_without_bfill_ffill():
    sp = _processor()
    n = 200
    acc = np.sin(np.linspace(0, 4 * np.pi, n))
    acc[50] = np.nan
    acc[51] = np.nan
    df = pd.DataFrame({
        "acc_x": acc,
        "acc_y": np.zeros(n),
        "acc_z": np.ones(n),
        "gyr_x": np.zeros(n),
    })
    out = sp._safe_filter(df)
    assert not out.empty
    assert out["acc_x"].isna().sum() == 0
    assert len(out) == n


def test_safe_filter_source_uses_linear_interpolate_only():
    import inspect

    source = inspect.getsource(SignalProcessor._safe_filter)
    assert "bfill" not in source
    assert "ffill" not in source
    assert 'method="linear"' in source or "method='linear'" in source
