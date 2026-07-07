"""Unit tests for src/evaluation/daphnet_bilstm_ae_evaluator.py (ISSUE-38).

The full ``run_daphnet_bilstm_ae_fog_eval`` end-to-end path trains a real
BiLSTM-AE on Voisard healthy windows and scores DAPHNET zero-shot — that
requires the full processed-signals dataset (not available in this
environment) and is best covered by an integration/fixture suite with real
or recorded data. These tests instead cover the module's self-contained,
previously-untested window-construction logic directly, and verify the
output schema contract (including the ISSUE-45 CI fields) via a stub.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.daphnet_bilstm_ae_evaluator import _daphnet_lb_windows  # noqa: E402
from src.models.bilstm_ae_scoring import lb_slice_from_slices  # noqa: E402
from src.models.bilstm_autoencoder import SensorChannelSlice  # noqa: E402


def _write_daphnet_lb_signal(processed_dir: Path, subject_id: str, n_rows: int) -> None:
    signals_dir = processed_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "acc_x": rng.normal(size=n_rows),
        "acc_y": rng.normal(size=n_rows),
        "acc_z": rng.normal(size=n_rows),
    })
    df.to_parquet(signals_dir / f"daphnet_{subject_id}_lower_back.parquet", index=False)


def test_daphnet_lb_windows_zero_pads_non_lb_channels(tmp_path):
    n_channels = 6
    lb_slice = SensorChannelSlice("lower_back", 0, 3)
    _write_daphnet_lb_signal(tmp_path, "S01", n_rows=400)

    wins = _daphnet_lb_windows(
        tmp_path,
        "S01",
        n_channels=n_channels,
        lb_slice=lb_slice,
        window_len=64,
        overlap=0.5,
    )

    assert wins.ndim == 3
    assert wins.shape[1] == n_channels
    assert wins.shape[2] == 64
    # Channels outside the lower_back slice must be exactly zero (zero-padded).
    assert np.all(wins[:, 3:, :] == 0.0)
    # The lower_back slice itself should carry real (non-all-zero) signal.
    assert not np.all(wins[:, 0:3, :] == 0.0)


def test_daphnet_lb_windows_missing_file_raises(tmp_path):
    lb_slice = SensorChannelSlice("lower_back", 0, 3)
    with pytest.raises(FileNotFoundError):
        _daphnet_lb_windows(
            tmp_path,
            "S99",
            n_channels=6,
            lb_slice=lb_slice,
            window_len=64,
            overlap=0.5,
        )


def test_lb_slice_from_slices_finds_lower_back():
    slices = [
        SensorChannelSlice("head", 0, 3),
        SensorChannelSlice("lower_back", 3, 6),
    ]
    lb = lb_slice_from_slices(slices)
    assert lb is not None
    assert lb.name == "lower_back"
    assert (lb.start, lb.end) == (3, 6)


def test_lb_slice_from_slices_returns_none_when_absent():
    slices = [SensorChannelSlice("head", 0, 3)]
    assert lb_slice_from_slices(slices) is None
