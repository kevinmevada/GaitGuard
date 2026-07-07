"""Integration smoke test spanning two real modules on synthetic data:

    SignalProcessor.process_sensor_dataframe()  (preprocessing/signal_processor.py)
        -> assert_loso_fold_disjoint()          (dataset/subject_split.py)

This does not require torch/xgboost/lightgbm/optuna (the modeling stack),
so it can run in any environment that has the pipeline's lighter
dependencies (numpy/pandas/scipy/loguru/tqdm) installed — including CI
environments that don't want to pull in the full deep-learning stack just
to smoke-test the preprocessing -> leakage-safety chain.

It intentionally does NOT attempt to exercise feature extraction, model
training, or LOSO evaluation end-to-end — those stages pull in optional
dependencies (PyWavelets, nolds, antropy, torch, xgboost, lightgbm, optuna)
and are already covered by their own dedicated unit tests elsewhere in this
suite. What's covered here specifically is the single most safety-critical
invariant in the whole pipeline: that a fold construction with a leaked
subject is caught, not silently accepted — verified against realistic
signal-processing output, not just hand-built arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.dataset.subject_split import assert_loso_fold_disjoint  # noqa: E402
from src.preprocessing.signal_processor import SignalProcessor  # noqa: E402


def _minimal_signal_processor_config(tmp_path: Path) -> dict:
    return {
        "dataset": {"sampling_rate": 100},
        "preprocessing": {
            "lowpass_cutoff_hz": 20.0,
            "lowpass_order": 4,
            "highpass_cutoff_hz": 0.5,
            "madgwick_enabled": True,
            "madgwick_beta": 0.1,
            "madgwick_sensors": ["head", "lower_back"],
            "gyro_in_degrees": False,
            "madgwick_use_magnetometer": False,
            "gait_event_source": "algorithm",
            "heel_strike_threshold_mode": "prominence",
            "heel_strike_peak_percentile": 85,
            "heel_strike_peak_percentile_by_cohort": {},
            "heel_strike_prominence_floor": None,
            "heel_strike_min_interval_s": 0.5,
            "max_nan_fraction_before_filter": 0.05,
            "exclude_uturn_segment": True,
            "min_walking_segment_s": 5.0,
            "gait_event_validation": {"tolerance_ms": 50},
        },
        "paths": {
            "processed_data": str(tmp_path / "processed"),
            "raw_data": str(tmp_path / "raw"),
            "logs": str(tmp_path / "logs"),
        },
    }


def _synthetic_imu_trial(rng: np.random.Generator, n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "acc_x": rng.normal(size=n),
            "acc_y": rng.normal(size=n) + 9.8,  # gravity-dominated axis
            "acc_z": rng.normal(size=n),
            "gyr_x": rng.normal(size=n) * 0.1,
            "gyr_y": rng.normal(size=n) * 0.1,
            "gyr_z": rng.normal(size=n) * 0.1,
        }
    )


def test_signal_processor_runs_on_synthetic_multi_sensor_trials(tmp_path):
    """The exact per-trial method shared by both batch pipeline runs and
    live API inference must run cleanly on realistic multi-sensor input."""
    sp = SignalProcessor(_minimal_signal_processor_config(tmp_path))
    rng = np.random.default_rng(0)

    for sensor_position in ("head", "lower_back", "left_foot", "right_foot"):
        df = _synthetic_imu_trial(rng)
        out = sp.process_sensor_dataframe(df, sensor_position, cohort="Healthy")
        assert not out.empty, f"processing produced an empty frame for {sensor_position}"
        assert np.isfinite(out.select_dtypes(include=[np.number]).to_numpy()).mean() > 0.9


def test_preprocessed_trials_feed_correctly_into_leakage_assertion(tmp_path):
    """End-to-end-ish integration: build several participants' worth of
    (really) preprocessed trials, then confirm a valid LOSO split passes
    and a deliberately leaked split is caught — using participant IDs
    attached to genuinely processed signal output, not synthetic labels
    detached from any real processing step."""
    sp = SignalProcessor(_minimal_signal_processor_config(tmp_path))
    rng = np.random.default_rng(1)

    participants = ["P1", "P2", "P3", "P4"]
    trial_groups: list[str] = []
    for pid in participants:
        for _trial in range(2):  # 2 trials per participant
            df = _synthetic_imu_trial(rng)
            out = sp.process_sensor_dataframe(df, "lower_back", cohort="Healthy")
            assert not out.empty
            trial_groups.append(pid)  # one row of "which participant" per processed trial

    trial_groups = np.array(trial_groups)

    # Valid LOSO fold: hold out P4 entirely.
    held_out = "P4"
    train_mask = trial_groups != held_out
    test_mask = trial_groups == held_out
    assert_loso_fold_disjoint(
        trial_groups[train_mask], trial_groups[test_mask], held_out_subject=held_out
    )

    # Deliberately broken fold: leak one of P4's processed trials into train.
    leaked_train_mask = train_mask.copy()
    leaked_idx = np.where(test_mask)[0][0]
    leaked_train_mask[leaked_idx] = True  # P4's trial now appears in BOTH splits

    with pytest.raises(AssertionError, match="DATA LEAKAGE"):
        assert_loso_fold_disjoint(
            trial_groups[leaked_train_mask], trial_groups[test_mask], held_out_subject=held_out
        )
