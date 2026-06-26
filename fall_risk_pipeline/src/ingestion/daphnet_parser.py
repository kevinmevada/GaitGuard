"""
Parse DAPHNET Parkinson's gait flat ``.txt`` recordings (no header row).

Column layout (whitespace-separated, 11 columns):
  0      — timestamp (ms)
  1–3    — ankle accelerometer (x, y, z)
  4–6    — thigh accelerometer (x, y, z)
  7–9    — trunk accelerometer (x, y, z)
  10     — annotation (0 = pre-experiment calibration → drop immediately)

The public corpus has 17 recordings across 10 subjects (``S01R01`` … ``S10R02``).
Recordings for the same subject are concatenated along the time axis after
calibration rows are removed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DAPHNET_FILENAME_RE = re.compile(r"^S(\d+)R(\d+)\.txt$", re.IGNORECASE)
DAPHNET_N_COLUMNS = 11
DAPHNET_SENSOR_CODES = ("ANK", "TH", "TR")

COL_TIMESTAMP = 0
COL_ANKLE_ACC = slice(1, 4)
COL_THIGH_ACC = slice(4, 7)
COL_TRUNK_ACC = slice(7, 10)
COL_ANNOTATION = 10

DAPHNET_ANNOTATION_CALIBRATION = 0
DAPHNET_RAW_ANNOTATIONS = {0, 1, 2}  # 0 dropped at parse; still valid in raw QC
DAPHNET_VALID_ANNOTATIONS = {1, 2}  # 1 = normal walking, 2 = freezing episode

DAPHNET_COLUMN_NAMES = [
    "timestamp_ms",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "thigh_acc_x",
    "thigh_acc_y",
    "thigh_acc_z",
    "trunk_acc_x",
    "trunk_acc_y",
    "trunk_acc_z",
    "annotation",
]

DAPHNET_SENSOR_PREFIXES = {
    "ankle": "ankle_acc",
    "thigh": "thigh_acc",
    "trunk": "trunk_acc",
}


@dataclass
class DaphnetFileStats:
    path: str
    trial_id: str
    subject_id: str
    n_rows_raw: int = 0
    n_rows_dropped_calibration: int = 0
    n_rows_kept: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "trial_id": self.trial_id,
            "subject_id": self.subject_id,
            "n_rows_raw": self.n_rows_raw,
            "n_rows_dropped_calibration": self.n_rows_dropped_calibration,
            "n_rows_kept": self.n_rows_kept,
        }


@dataclass
class DaphnetSubjectBundle:
    subject_id: str
    frame: pd.DataFrame
    source_trials: list[str] = field(default_factory=list)
    file_stats: list[DaphnetFileStats] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)


def subject_id_from_path(path: Path) -> str:
    match = DAPHNET_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a DAPHNET recording filename: {path.name}")
    return f"S{match.group(1)}"


def trial_id_from_path(path: Path) -> str:
    match = DAPHNET_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a DAPHNET recording filename: {path.name}")
    return path.stem.upper()


def discover_daphnet_files(root: Path) -> list[Path]:
    """Return all ``SxxRyy.txt`` files under ``root`` (sorted)."""
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.rglob("*.txt")
        if p.is_file() and DAPHNET_FILENAME_RE.match(p.name)
    )


def read_daphnet_txt_raw(path: Path) -> pd.DataFrame:
    """Load flat DAPHNET file with hardcoded column positions (no header)."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    if raw.empty:
        raise ValueError(f"DAPHNET file is empty: {path}")
    if raw.shape[1] != DAPHNET_N_COLUMNS:
        raise ValueError(
            f"DAPHNET {path.name}: expected {DAPHNET_N_COLUMNS} columns, got {raw.shape[1]}"
        )
    if raw.isna().any().any():
        raise ValueError(f"DAPHNET {path.name}: non-numeric or missing values")
    raw.columns = DAPHNET_COLUMN_NAMES
    return raw


def parse_daphnet_file(
    path: Path,
    *,
    drop_calibration: bool = True,
) -> tuple[pd.DataFrame, DaphnetFileStats]:
    """
    Parse one recording; drop ``annotation == 0`` calibration rows by default.
    """
    path = Path(path)
    raw = read_daphnet_txt_raw(path)
    n_raw = len(raw)

    if drop_calibration:
        kept = raw.loc[raw["annotation"] != DAPHNET_ANNOTATION_CALIBRATION].copy()
    else:
        kept = raw.copy()

    kept = kept.reset_index(drop=True)
    stats = DaphnetFileStats(
        path=str(path),
        trial_id=trial_id_from_path(path),
        subject_id=subject_id_from_path(path),
        n_rows_raw=n_raw,
        n_rows_dropped_calibration=n_raw - len(kept),
        n_rows_kept=len(kept),
    )
    return kept, stats


def concat_daphnet_recordings(
    frames: list[pd.DataFrame],
    trial_ids: list[str],
) -> pd.DataFrame:
    """Concatenate per-recording frames for one subject (preserves file order)."""
    if not frames:
        return pd.DataFrame(columns=[*DAPHNET_COLUMN_NAMES, "source_trial"])
    parts: list[pd.DataFrame] = []
    for trial_id, frame in zip(trial_ids, frames):
        part = frame.copy()
        part["source_trial"] = trial_id
        parts.append(part)
    return pd.concat(parts, axis=0, ignore_index=True)


def group_daphnet_files(paths: list[Path]) -> dict[str, list[Path]]:
    """Group recording paths by subject id (``S01``, ``S02``, …)."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        if not DAPHNET_FILENAME_RE.match(path.name):
            continue
        grouped[subject_id_from_path(path)].append(path)

    for subject_id in grouped:
        grouped[subject_id] = sorted(
            grouped[subject_id],
            key=lambda p: int(DAPHNET_FILENAME_RE.match(p.name).group(2)),  # type: ignore[union-attr]
        )
    return dict(sorted(grouped.items()))


def load_daphnet_per_subject(
    root: Path,
    *,
    drop_calibration: bool = True,
) -> dict[str, DaphnetSubjectBundle]:
    """
    Parse all DAPHNET flat files under ``root`` and concatenate per subject.

    Returns one ``DaphnetSubjectBundle`` per subject with calibration rows removed.
    """
    paths = discover_daphnet_files(root)
    grouped = group_daphnet_files(paths)
    bundles: dict[str, DaphnetSubjectBundle] = {}

    for subject_id, subject_paths in grouped.items():
        frames: list[pd.DataFrame] = []
        stats_list: list[DaphnetFileStats] = []
        trial_ids: list[str] = []

        for path in subject_paths:
            frame, stats = parse_daphnet_file(path, drop_calibration=drop_calibration)
            frames.append(frame)
            stats_list.append(stats)
            trial_ids.append(stats.trial_id)

        bundles[subject_id] = DaphnetSubjectBundle(
            subject_id=subject_id,
            frame=concat_daphnet_recordings(frames, trial_ids),
            source_trials=trial_ids,
            file_stats=stats_list,
        )

    return bundles


def daphnet_frame_to_sensor_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split a wide DAPHNET frame into per-sensor IMU tables (``time``, ``acc_*``).

    ``time`` is seconds relative to the first kept row (after calibration drop).
    """
    if frame.empty:
        return {
            sensor: pd.DataFrame(columns=["time", "acc_x", "acc_y", "acc_z"])
            for sensor in DAPHNET_SENSOR_PREFIXES
        }

    t0 = float(frame["timestamp_ms"].iloc[0])
    time_s = (frame["timestamp_ms"].astype(np.float64) - t0) / 1000.0
    signals: dict[str, pd.DataFrame] = {}

    for sensor, prefix in DAPHNET_SENSOR_PREFIXES.items():
        signals[sensor] = pd.DataFrame(
            {
                "time": time_s,
                "acc_x": frame[f"{prefix}_x"].astype(np.float32),
                "acc_y": frame[f"{prefix}_y"].astype(np.float32),
                "acc_z": frame[f"{prefix}_z"].astype(np.float32),
            }
        )
    return signals


def ingest_summary_rows(bundles: dict[str, DaphnetSubjectBundle]) -> list[dict[str, Any]]:
    """Flatten per-file parse stats for CSV export."""
    rows: list[dict[str, Any]] = []
    for bundle in bundles.values():
        for stats in bundle.file_stats:
            row = stats.to_dict()
            row["n_rows_subject_concat"] = bundle.n_rows
            rows.append(row)
    return rows
