"""
Parse Voisard Figshare tab-separated IMU ``*_raw_data_{HE,LB,LF,RF}.txt`` files.

- Reads header row (PacketCounter, Acc_*, Gyr_*, Mag_*)
- Converts gyroscope deg/s → rad/s at parse time
- Detects PacketCounter gaps; interpolates small gaps or truncates at large ones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

VOISARD_TXT_COLUMNS = [
    "PacketCounter",
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
    "Mag_X",
    "Mag_Y",
    "Mag_Z",
]

CANONICAL_IMU_COLUMNS = [
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

GYR_CANONICAL = ("gyr_x", "gyr_y", "gyr_z")
DEG_TO_RAD = np.pi / 180.0

GapStrategy = Literal["interpolate", "truncate"]


@dataclass
class PacketGapInfo:
    sensor_path: str = ""
    n_gaps: int = 0
    gap_indices: list[int] = field(default_factory=list)
    gap_details: list[str] = field(default_factory=list)
    max_gap: int = 0
    repaired: bool = False
    truncated: bool = False
    n_rows_before: int = 0
    n_rows_after: int = 0


def is_packet_counter_wrap(previous: int, current: int) -> bool:
    """Allow uint16 PacketCounter rollover (e.g. 65535 → 0)."""
    return previous >= 60_000 and current <= 1_000


def find_packet_counter_gaps(counter: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Return list of (row_index, prev_counter, next_counter, diff) for non-unit steps.
    """
    values = counter.astype(np.int64)
    if len(values) < 2:
        return []
    gaps: list[tuple[int, int, int, int]] = []
    for idx, diff in enumerate(np.diff(values)):
        d = int(diff)
        if d == 1:
            continue
        if d < 0 and is_packet_counter_wrap(int(values[idx]), int(values[idx + 1])):
            continue
        gaps.append((idx, int(values[idx]), int(values[idx + 1]), d))
    return gaps


def read_voisard_txt_raw(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Load tab-separated Voisard file with canonical column names (gyro still deg/s)."""
    issues: list[str] = []
    try:
        preview = pd.read_csv(path, sep=r"\s+", nrows=1)
        has_header = (
            preview.shape[1] == len(VOISARD_TXT_COLUMNS)
            and str(preview.columns[0]).strip() == "PacketCounter"
        )
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=0 if has_header else None,
            names=None if has_header else VOISARD_TXT_COLUMNS,
        )
    except Exception as exc:
        return None, [str(exc)]

    if df.shape[1] != len(VOISARD_TXT_COLUMNS):
        issues.append(f"expected {len(VOISARD_TXT_COLUMNS)} columns, got {df.shape[1]}")
        return None, issues

    df.columns = VOISARD_TXT_COLUMNS
    numeric = df.apply(pd.to_numeric, errors="coerce")
    corrupted = int(numeric.isna().any(axis=1).sum())
    if corrupted:
        issues.append(f"{corrupted} non-numeric rows")
        return None, issues

    return numeric.dropna().reset_index(drop=True), issues


def _interpolate_gap(
    df: pd.DataFrame,
    gap_index: int,
    diff: int,
) -> pd.DataFrame:
    """Insert ``diff - 1`` linearly interpolated rows after ``gap_index``."""
    if diff <= 1:
        return df

    n_insert = diff - 1
    row_a = df.iloc[gap_index]
    row_b = df.iloc[gap_index + 1]
    imu_cols = [c for c in VOISARD_TXT_COLUMNS if c != "PacketCounter"]

    new_rows: list[dict] = []
    base_counter = int(row_a["PacketCounter"])
    for j in range(1, n_insert + 1):
        alpha = j / diff
        row: dict = {"PacketCounter": base_counter + j}
        for col in imu_cols:
            row[col] = (1.0 - alpha) * float(row_a[col]) + alpha * float(row_b[col])
        new_rows.append(row)

    top = df.iloc[: gap_index + 1]
    bottom = df.iloc[gap_index + 1 :]
    inserted = pd.DataFrame(new_rows, columns=VOISARD_TXT_COLUMNS)
    return pd.concat([top, inserted, bottom], ignore_index=True)


def _repair_packet_gaps(
    df: pd.DataFrame,
    *,
    gap_strategy: GapStrategy,
    max_interpolate_gap: int,
) -> tuple[pd.DataFrame, PacketGapInfo]:
    info = PacketGapInfo(n_rows_before=len(df))
    gaps = find_packet_counter_gaps(df["PacketCounter"].to_numpy())

    if not gaps:
        info.n_rows_after = len(df)
        return df, info

    info.n_gaps = len(gaps)
    info.gap_indices = [g[0] for g in gaps]
    info.max_gap = max(g[3] for g in gaps)
    info.gap_details = [
        f"rows {i}->{i + 1}: {prev} to {nxt} (diff={diff})"
        for i, prev, nxt, diff in gaps
    ]

    if gap_strategy == "truncate":
        first_idx = gaps[0][0]
        first_diff = gaps[0][3]
        if first_diff > max_interpolate_gap:
            df = df.iloc[: first_idx + 1].reset_index(drop=True)
            info.truncated = True
            info.repaired = True
            info.n_rows_after = len(df)
            return df, info

    # Interpolate small gaps; re-scan after each insert (indices shift).
    repaired = False
    while True:
        gaps = find_packet_counter_gaps(df["PacketCounter"].to_numpy())
        if not gaps:
            break
        idx, _prev, _nxt, diff = gaps[0]
        if diff > max_interpolate_gap:
            if gap_strategy == "truncate":
                df = df.iloc[: idx + 1].reset_index(drop=True)
                info.truncated = True
                repaired = True
            break
        df = _interpolate_gap(df, idx, diff)
        repaired = True

    info.repaired = repaired
    info.n_rows_after = len(df)
    return df, info


def voisard_txt_to_imu_frame(
    path: Path,
    *,
    fs: float = 100.0,
    convert_gyro_to_rad: bool = True,
    gap_strategy: GapStrategy = "interpolate",
    max_interpolate_gap: int = 10,
) -> tuple[pd.DataFrame | None, list[str], PacketGapInfo]:
    """
    Parse one Voisard sensor ``.txt`` into a canonical IMU DataFrame.

    Output columns: ``time``, ``acc_x`` … ``mag_z`` (gyro in rad/s when converted).
    ``time`` is uniform ``arange(n) / fs`` after gap repair.
    """
    gap_info = PacketGapInfo(sensor_path=str(path))
    raw, issues = read_voisard_txt_raw(path)
    if raw is None:
        return None, issues, gap_info

    raw, gap_info = _repair_packet_gaps(
        raw,
        gap_strategy=gap_strategy,
        max_interpolate_gap=max_interpolate_gap,
    )
    if len(raw) < 2:
        return None, ["too few rows after gap repair"], gap_info

    rename = {
        "Acc_X": "acc_x",
        "Acc_Y": "acc_y",
        "Acc_Z": "acc_z",
        "Gyr_X": "gyr_x",
        "Gyr_Y": "gyr_y",
        "Gyr_Z": "gyr_z",
        "Mag_X": "mag_x",
        "Mag_Y": "mag_y",
        "Mag_Z": "mag_z",
    }
    imu = raw.drop(columns=["PacketCounter"]).rename(columns=rename)

    if convert_gyro_to_rad:
        for col in GYR_CANONICAL:
            imu[col] = imu[col].astype(float) * DEG_TO_RAD

    imu.insert(0, "time", np.arange(len(imu), dtype=float) / float(fs))
    return imu.reset_index(drop=True), [], gap_info
