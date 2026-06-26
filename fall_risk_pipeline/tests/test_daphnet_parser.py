"""Tests for DAPHNET flat-file parsing (hardcoded columns, calibration drop)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.daphnet_parser import (
    DAPHNET_COLUMN_NAMES,
    DAPHNET_N_COLUMNS,
    concat_daphnet_recordings,
    daphnet_frame_to_sensor_signals,
    discover_daphnet_files,
    group_daphnet_files,
    load_daphnet_per_subject,
    parse_daphnet_file,
    read_daphnet_txt_raw,
)


def _daphnet_row(
    timestamp: float,
    annotation: int,
    *,
    ankle: tuple[float, float, float] = (1.0, 2.0, 3.0),
    thigh: tuple[float, float, float] = (4.0, 5.0, 6.0),
    trunk: tuple[float, float, float] = (7.0, 8.0, 9.0),
) -> str:
    ax, ay, az = ankle
    tx, ty, tz = thigh
    ux, uy, uz = trunk
    return (
        f"{timestamp} {ax} {ay} {az} {tx} {ty} {tz} {ux} {uy} {uz} {annotation}"
    )


def test_read_daphnet_hardcoded_columns(tmp_path: Path):
    path = tmp_path / "S01R01.txt"
    path.write_text(
        "\n".join(
            [
                _daphnet_row(15, 0),
                _daphnet_row(31, 1),
                _daphnet_row(47, 2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    raw = read_daphnet_txt_raw(path)
    assert list(raw.columns) == DAPHNET_COLUMN_NAMES
    assert raw.shape == (3, DAPHNET_N_COLUMNS)
    assert raw.iloc[0]["timestamp_ms"] == 15
    assert raw.iloc[0]["ankle_acc_x"] == 1.0
    assert raw.iloc[0]["trunk_acc_z"] == 9.0
    assert raw.iloc[0]["annotation"] == 0


def test_parse_drops_annotation_zero(tmp_path: Path):
    path = tmp_path / "S01R01.txt"
    path.write_text(
        "\n".join(
            [
                _daphnet_row(15, 0),
                _daphnet_row(31, 0),
                _daphnet_row(47, 1),
                _daphnet_row(63, 2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frame, stats = parse_daphnet_file(path)
    assert stats.n_rows_raw == 4
    assert stats.n_rows_dropped_calibration == 2
    assert stats.n_rows_kept == 2
    assert set(frame["annotation"].astype(int)) == {1, 2}
    assert (frame["annotation"] == 0).sum() == 0


def test_wrong_column_count_raises(tmp_path: Path):
    path = tmp_path / "S01R01.txt"
    path.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 11 columns"):
        read_daphnet_txt_raw(path)


def _write_corpus_17_files(root: Path) -> None:
    """10 subjects; 7 with two recordings → 17 files total."""
    subjects_with_two = {1, 2, 3, 4, 5, 6, 7}
    for s in range(1, 11):
        n_recs = 2 if s in subjects_with_two else 1
        for r in range(1, n_recs + 1):
            path = root / f"S{s:02d}R0{r}.txt"
            lines = [
                _daphnet_row(10, 0),
                _daphnet_row(10 + r * 100, 1),
                _daphnet_row(10 + r * 100 + 16, 1),
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_discover_and_group_17_files(tmp_path: Path):
    root = tmp_path / "daphnet"
    root.mkdir()
    _write_corpus_17_files(root)

    files = discover_daphnet_files(root)
    assert len(files) == 17

    grouped = group_daphnet_files(files)
    assert len(grouped) == 10
    assert len(grouped["S01"]) == 2
    assert len(grouped["S08"]) == 1
    assert grouped["S01"][0].name == "S01R01.txt"
    assert grouped["S01"][1].name == "S01R02.txt"


def test_concat_per_subject_drops_calibration(tmp_path: Path):
    root = tmp_path / "daphnet"
    root.mkdir()
    _write_corpus_17_files(root)

    bundles = load_daphnet_per_subject(root)
    assert len(bundles) == 10

    s01 = bundles["S01"]
    assert s01.source_trials == ["S01R01", "S01R02"]
    assert len(s01.file_stats) == 2
    assert all(s.n_rows_dropped_calibration == 1 for s in s01.file_stats)
    assert (s01.frame["annotation"] == 0).sum() == 0
    assert s01.n_rows == 4  # 2 kept rows × 2 recordings
    assert set(s01.frame["source_trial"]) == {"S01R01", "S01R02"}

    total_raw = 17 * 3
    total_dropped = 17
    total_kept = total_raw - total_dropped
    assert sum(b.n_rows for b in bundles.values()) == total_kept


def test_sensor_signals_time_relative(tmp_path: Path):
    path = tmp_path / "S03R01.txt"
    path.write_text(
        "\n".join([_daphnet_row(1000, 0), _daphnet_row(1016, 1), _daphnet_row(1032, 1)])
        + "\n",
        encoding="utf-8",
    )
    frame, _ = parse_daphnet_file(path)
    signals = daphnet_frame_to_sensor_signals(frame)
    assert set(signals) == {"ankle", "thigh", "trunk"}
    trunk = signals["trunk"]
    assert np.isclose(trunk["time"].iloc[0], 0.0)
    assert np.isclose(trunk["time"].iloc[1], 0.016)
    assert list(trunk.columns) == ["time", "acc_x", "acc_y", "acc_z"]


def test_concat_preserves_order(tmp_path: Path):
    f1 = pd.DataFrame(
        {
            "timestamp_ms": [1.0, 2.0],
            "ankle_acc_x": [1.0, 1.0],
            "ankle_acc_y": [0.0, 0.0],
            "ankle_acc_z": [0.0, 0.0],
            "thigh_acc_x": [0.0, 0.0],
            "thigh_acc_y": [0.0, 0.0],
            "thigh_acc_z": [0.0, 0.0],
            "trunk_acc_x": [0.0, 0.0],
            "trunk_acc_y": [0.0, 0.0],
            "trunk_acc_z": [0.0, 0.0],
            "annotation": [1, 1],
        }
    )
    f2 = f1.copy()
    f2["timestamp_ms"] = [10.0, 11.0]
    out = concat_daphnet_recordings([f1, f2], ["S01R01", "S01R02"])
    assert out["source_trial"].tolist() == ["S01R01", "S01R01", "S01R02", "S01R02"]
