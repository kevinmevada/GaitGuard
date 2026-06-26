"""U-turn exclusion from walking signals (straight-bout concatenation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.walking_segments import extract_walking_segments


def _synthetic_trial(n: int = 2000, fs: float = 100.0) -> pd.DataFrame:
    t = np.arange(n) / fs
    return pd.DataFrame(
        {
            "time": t,
            "acc_x": np.ones(n),
            "acc_z": np.linspace(0, 1, n),
            "gyr_z": np.zeros(n),
        }
    )


def test_extract_walking_segments_concatenates_outward_and_return():
    df = _synthetic_trial(2000)
    walking, info = extract_walking_segments(df, 800, 1200, fs=100.0, min_segment_s=5.0)
    assert info["status"] == "ok"
    assert info["outward_samples"] == 800
    assert info["return_samples"] == 800
    assert info["excluded_samples"] == 400
    assert len(walking) == 1600
    assert walking["time"].iloc[-1] == pytest.approx(15.99, abs=0.05)


def test_rejects_short_segment():
    df = _synthetic_trial(600)
    walking, info = extract_walking_segments(df, 200, 250, fs=100.0, min_segment_s=5.0)
    assert walking is None
    assert info["status"] == "segment_too_short"


def test_rejects_invalid_bounds():
    df = _synthetic_trial(1000)
    walking, info = extract_walking_segments(df, 900, 800, fs=100.0)
    assert walking is None
    assert info["status"] == "invalid_bounds"


def test_signal_processor_excludes_uturn(tmp_path, monkeypatch):
    from src.preprocessing.signal_processor import SignalProcessor

    proc = tmp_path / "processed"
    signals = proc / "signals"
    signals.mkdir(parents=True)
    meta_path = proc / "trial_metadata.csv"
    pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "cohort": "Healthy",
                "uturn_start": 500,
                "uturn_end": 700,
            }
        ]
    ).to_csv(meta_path, index=False)

    n = 1500
    df = _synthetic_trial(n)
    for pos in ["head", "lower_back", "left_foot", "right_foot"]:
        df.to_parquet(signals / f"T1_{pos}.parquet", index=False)

    cfg = {
        "paths": {
            "processed_data": str(proc),
            "metrics": str(tmp_path / "metrics"),
        },
        "dataset": {"sampling_rate": 100},
        "preprocessing": {
            "lowpass_cutoff_hz": 15,
            "highpass_cutoff_hz": 0.1,
            "lowpass_order": 4,
            "madgwick_enabled": False,
            "madgwick_beta": 0.1,
            "gyro_in_degrees": False,
            "gait_event_source": "algorithm",
            "exclude_uturn_segment": True,
            "min_walking_segment_s": 5.0,
        },
    }
    SignalProcessor(cfg).run()
    out = pd.read_parquet(proc / "signals_clean" / "T1_lower_back.parquet")
    assert len(out) == 1300  # 500 + 800
