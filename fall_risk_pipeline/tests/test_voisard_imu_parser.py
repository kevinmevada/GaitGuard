"""Voisard IMU .txt parsing: columns, gyro rad/s, PacketCounter gaps."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.voisard_imu_parser import (
    DEG_TO_RAD,
    find_packet_counter_gaps,
    is_packet_counter_wrap,
    voisard_txt_to_imu_frame,
)


def _write_voisard_txt(
    path: Path,
    *,
    n_rows: int = 100,
    packet_start: int = 1000,
    gap_at: int | None = None,
    gap_size: int = 5,
    gyr_deg: float = 90.0,
) -> None:
    lines = [
        "PacketCounter\tAcc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z"
    ]
    counter = packet_start
    for i in range(n_rows):
        if gap_at is not None and i == gap_at:
            counter += gap_size - 1
        lines.append(
            f"{counter}\t1.0\t2.0\t9.81\t{gyr_deg}\t0.0\t0.0\t1.0\t2.0\t3.0"
        )
        counter += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parses_header_and_canonical_columns(tmp_path: Path):
    path = tmp_path / "HS_1_1_raw_data_HE.txt"
    _write_voisard_txt(path)
    df, issues, gap = voisard_txt_to_imu_frame(path, fs=100.0)
    assert not issues
    assert gap.n_gaps == 0
    assert list(df.columns) == [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ]
    assert len(df) == 100
    assert df["acc_z"].iloc[0] == pytest.approx(9.81)


def test_gyro_converted_to_rad_at_parse_time(tmp_path: Path):
    path = tmp_path / "sensor.txt"
    _write_voisard_txt(path, gyr_deg=90.0)
    df, _, _ = voisard_txt_to_imu_frame(path, convert_gyro_to_rad=True)
    assert df["gyr_x"].iloc[0] == pytest.approx(90.0 * DEG_TO_RAD)
    assert df["gyr_x"].iloc[0] == pytest.approx(math.pi / 2)


def test_packet_counter_gap_detected_and_interpolated(tmp_path: Path):
    path = tmp_path / "sensor.txt"
    _write_voisard_txt(path, n_rows=20, gap_at=5, gap_size=4)
    df, issues, gap = voisard_txt_to_imu_frame(
        path, gap_strategy="interpolate", max_interpolate_gap=10
    )
    assert not issues
    assert gap.n_gaps >= 1
    assert gap.repaired
    assert len(df) == 23  # diff=4 → 3 interpolated rows (20 + 3)


def test_large_gap_truncates(tmp_path: Path):
    path = tmp_path / "sensor.txt"
    _write_voisard_txt(path, n_rows=50, gap_at=10, gap_size=20)
    df, issues, gap = voisard_txt_to_imu_frame(
        path, gap_strategy="truncate", max_interpolate_gap=5
    )
    assert not issues
    assert gap.truncated
    assert len(df) == 10  # rows 0..9 before first large gap


def test_uint16_packet_counter_wrap_allowed():
    assert is_packet_counter_wrap(65535, 0)
    gaps = find_packet_counter_gaps(np.array([65534, 65535, 0, 1], dtype=np.int64))
    assert gaps == []


def test_find_packet_counter_gap():
    gaps = find_packet_counter_gaps(np.array([10, 11, 16, 17], dtype=np.int64))
    assert len(gaps) == 1
    assert gaps[0][3] == 5
