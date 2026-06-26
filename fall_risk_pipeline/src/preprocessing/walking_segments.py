"""
Exclude annotated U-turn segments from walking analysis signals.

Figshare trials are: outward walk → 180° U-turn → return walk. Stop-start
dynamics in the turn segment must not enter healthy-reference models or stride
features; only concatenated straight walking bouts are retained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def extract_walking_segments(
    df: pd.DataFrame,
    uturn_start: int,
    uturn_end: int,
    *,
    fs: float = 100.0,
    min_segment_s: float = 5.0,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Keep outward + return straight-walking bouts; drop the U-turn between them.

    outward = signal[:uturn_start]
    return_ = signal[uturn_end:]
    walking = concat(outward, return_)

    Each retained segment must be >= ``min_segment_s`` (default 5 s → 500 @ 100 Hz).
    """
    info: dict[str, Any] = {
        "uturn_start": int(uturn_start),
        "uturn_end": int(uturn_end),
        "n_samples_in": len(df),
        "min_segment_samples": int(min_segment_s * fs),
    }

    if df.empty:
        info["status"] = "empty_input"
        return None, info

    n = len(df)
    if uturn_end <= uturn_start or uturn_start < 0 or uturn_end > n:
        info["status"] = "invalid_bounds"
        return None, info

    outward = df.iloc[:uturn_start]
    return_seg = df.iloc[uturn_end:]
    min_samples = int(min_segment_s * fs)

    info["outward_samples"] = len(outward)
    info["return_samples"] = len(return_seg)
    info["excluded_samples"] = int(uturn_end - uturn_start)

    if len(outward) < min_samples or len(return_seg) < min_samples:
        info["status"] = "segment_too_short"
        return None, info

    walking = pd.concat([outward, return_seg], ignore_index=True)
    if "time" in walking.columns:
        walking = walking.copy()
        walking["time"] = np.arange(len(walking), dtype=float) / float(fs)

    info["n_samples_out"] = len(walking)
    info["status"] = "ok"
    return walking, info
