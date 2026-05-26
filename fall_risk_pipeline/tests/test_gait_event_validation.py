"""Unit tests for gait-event ground-truth parsing and matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.gait_events_gt import (
    gait_events_from_metadata,
    heel_strike_indices,
    load_ground_truth_gait_events,
    match_heel_strikes,
)
from src.preprocessing.signal_processor import SignalProcessor


@pytest.fixture
def processor_config() -> dict:
    return {
        "paths": {"processed_data": "data/processed", "raw_data": "data/raw", "metrics": "results/metrics"},
        "dataset": {"sampling_rate": 100},
        "preprocessing": {
            "lowpass_cutoff_hz": 15.0,
            "lowpass_order": 4,
            "highpass_cutoff_hz": 0.1,
            "madgwick_beta": 0.1,
            "gait_event_source": "algorithm",
        },
    }


def test_metadata_to_heel_strike_indices():
    meta = {
        "leftGaitEvents": [[100, 150], [300, 360]],
        "rightGaitEvents": [[110, 170]],
    }
    events = gait_events_from_metadata(meta)
    assert events is not None
    left_hs = heel_strike_indices(events, side="left")
    right_hs = heel_strike_indices(events, side="right")
    assert list(left_hs) == [150, 360]
    assert list(right_hs) == [170]


def test_match_heel_strikes_precision_recall():
    gt = np.array([100, 200, 300])
    det = np.array([102, 198, 400, 500])  # 2 TP, 2 FP, 1 FN
    m = match_heel_strikes(det, gt, tolerance_samples=5)
    assert m["tp"] == 2
    assert m["fp"] == 2
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)


def test_detect_heel_strikes_on_synthetic_peaks(processor_config):
    fs = 100
    n = 2000
    t = np.arange(n) / fs
    acc_z = -9.8 * np.ones(n)
    # Heel-strike peaks are maxima of -acc_z → local minima (impulses) on acc_z
    strike_times = np.arange(0.5, 18.0, 1.0)
    for st in strike_times:
        idx = int(st * fs)
        if idx < n:
            acc_z[idx : idx + 5] -= 6.0

    df = pd.DataFrame({
        "time": t,
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": acc_z,
        "gyr_x": np.zeros(n),
        "gyr_y": np.zeros(n),
        "gyr_z": np.zeros(n),
    })
    sp = SignalProcessor(processor_config)
    detected = sp.detect_heel_strike_indices(df, "left")
    expected = (strike_times * fs).astype(int)
    m = match_heel_strikes(detected, expected, tolerance_samples=int(0.05 * fs))
    assert m["recall"] >= 0.8
    assert m["precision"] >= 0.5


def test_load_ground_truth_from_meta_json(tmp_path: Path):
    trial_dir = tmp_path / "HS_2_1"
    trial_dir.mkdir()
    meta = {
        "leftGaitEvents": [[50, 100]],
        "rightGaitEvents": [[55, 105]],
        "freq": 100,
    }
    (trial_dir / "HS_2_1_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    gt = load_ground_truth_gait_events(trial_dir)
    assert gt is not None
    assert heel_strike_indices(gt, side="left").tolist() == [100]
