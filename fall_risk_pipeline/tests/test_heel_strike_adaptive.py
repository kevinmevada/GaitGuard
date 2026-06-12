"""Tests for adaptive heel-strike detection thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.gait_events_gt import match_heel_strikes, resolve_trial_cohort
from src.preprocessing.signal_processor import SignalProcessor


def _synthetic_foot_df(
    *,
    fs: int = 100,
    n: int = 2000,
    impulse_depth: float = 6.0,
    strike_interval_s: float = 1.0,
) -> pd.DataFrame:
    t = np.arange(n) / fs
    acc_z = -9.8 * np.ones(n)
    strike_times = np.arange(0.5, 18.0, strike_interval_s)
    for st in strike_times:
        idx = int(st * fs)
        if idx < n:
            acc_z[idx : idx + 5] -= impulse_depth
    return pd.DataFrame({
        "time": t,
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": acc_z,
        "gyr_x": np.zeros(n),
        "gyr_y": np.zeros(n),
        "gyr_z": np.zeros(n),
    })


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
            "heel_strike_threshold_mode": "prominence",
            "heel_strike_peak_percentile": 85,
            "heel_strike_min_interval_s": 0.5,
        },
    }


def test_prominence_mode_less_sensitive_to_amplitude_than_percentile(processor_config):
    fs = 100
    expected = (np.arange(0.5, 18.0, 1.0) * fs).astype(int)
    tol = int(0.05 * fs)

    high_df = _synthetic_foot_df(impulse_depth=8.0)
    low_df = _synthetic_foot_df(impulse_depth=2.5)

    prom_sp = SignalProcessor(processor_config)
    high_prom = prom_sp.detect_heel_strike_indices(high_df, "left")
    low_prom = prom_sp.detect_heel_strike_indices(low_df, "left")

    pct_cfg = {
        **processor_config,
        "preprocessing": {
            **processor_config["preprocessing"],
            "heel_strike_threshold_mode": "percentile",
        },
    }
    pct_sp = SignalProcessor(pct_cfg)
    high_pct = pct_sp.detect_heel_strike_indices(high_df, "left")
    low_pct = pct_sp.detect_heel_strike_indices(low_df, "left")

    low_prom_recall = match_heel_strikes(low_prom, expected, tol)["recall"]
    low_pct_recall = match_heel_strikes(low_pct, expected, tol)["recall"]
    assert low_prom_recall >= low_pct_recall

    high_overcount = abs(len(high_pct) - len(expected))
    high_prom_overcount = abs(len(high_prom) - len(expected))
    assert high_prom_overcount <= high_overcount


def test_cohort_specific_percentile_override(processor_config):
    cfg = {
        **processor_config,
        "preprocessing": {
            **processor_config["preprocessing"],
            "heel_strike_peak_percentile": 95,
            "heel_strike_peak_percentile_by_cohort": {"PD": 70},
        },
    }
    sp = SignalProcessor(cfg)
    assert sp._peak_percentile_for_cohort("PD") == 70
    assert sp._peak_percentile_for_cohort("Healthy") == 95


def test_resolve_trial_cohort_from_pathology_key(tmp_path):
    trial_dir = tmp_path / "PD" / "g1" / "s1" / "PD_1_1"
    trial_dir.mkdir(parents=True)
    meta = {"pathologyKey": "PD", "participant_id": "s1"}
    assert resolve_trial_cohort(trial_dir, meta, raw_dir=tmp_path) == "PD"
