"""
Ground-truth gait events from the Figshare clinical gait dataset.

Dataset reference (Voisard et al., Scientific Data 2025):
  - ``gait_events.csv`` per trial when exported, or
  - ``*_meta.json`` fields ``leftGaitEvents`` / ``rightGaitEvents``:
    each stride is [toe_off_sample, heel_strike_sample].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_trial_cohort(
    trial_dir: Path,
    meta: dict[str, Any],
    *,
    raw_dir: Path | None = None,
) -> str:
    """Resolve pathology cohort from metadata JSON or raw-data directory layout."""
    from src.ingestion.data_loader import PATHOLOGY_KEY_MAP

    if meta.get("cohort"):
        return str(meta["cohort"])
    pkey = meta.get("pathologyKey", "")
    if pkey in PATHOLOGY_KEY_MAP:
        return PATHOLOGY_KEY_MAP[pkey]
    if raw_dir is not None:
        try:
            rel = trial_dir.relative_to(raw_dir)
            if rel.parts:
                token = str(rel.parts[0])
                return PATHOLOGY_KEY_MAP.get(token, token)
        except ValueError:
            pass
    return "Unknown"


def load_trial_metadata_json(trial_dir: Path) -> dict[str, Any]:
    for path in sorted(trial_dir.glob("*_meta.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            continue
    return {}


def gait_events_from_metadata(meta: dict[str, Any]) -> pd.DataFrame | None:
    """Flatten leftGaitEvents / rightGaitEvents into a long-form table."""
    rows: list[dict[str, Any]] = []
    for side, key in (("left", "leftGaitEvents"), ("right", "rightGaitEvents")):
        pairs = meta.get(key)
        if pairs is None:
            continue
        for stride_idx, pair in enumerate(pairs):
            if pair is None or len(pair) < 2:
                continue
            rows.append({
                "side": side,
                "event": "toe_off",
                "sample_index": int(pair[0]),
                "stride_index": stride_idx,
            })
            rows.append({
                "side": side,
                "event": "heel_strike",
                "sample_index": int(pair[1]),
                "stride_index": stride_idx,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_ground_truth_gait_events(trial_dir: Path) -> pd.DataFrame | None:
    """Load ``gait_events.csv`` or build from trial metadata JSON."""
    events_path = trial_dir / "gait_events.csv"
    if events_path.exists():
        df = pd.read_csv(events_path)
        return _normalize_gait_events_df(df)

    meta = load_trial_metadata_json(trial_dir)
    if meta:
        return gait_events_from_metadata(meta)
    return None


def _normalize_gait_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for heel-strike extraction."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def heel_strike_indices(
    gait_events: pd.DataFrame | None,
    *,
    side: str,
    n_samples: int | None = None,
) -> np.ndarray:
    """
    Return sorted sample indices of ground-truth heel strikes for ``left`` or ``right``.
    """
    if gait_events is None or gait_events.empty:
        return np.array([], dtype=int)

    df = _normalize_gait_events_df(gait_events)
    side = side.lower()
    side_aliases = {side, "l" if side == "left" else "r", f"{side[0]}f", f"{side}_foot"}
    if side == "left":
        side_aliases |= {"lf", "left_foot"}
    else:
        side_aliases |= {"rf", "right_foot"}

    idx: list[int] = []

    if "event" in df.columns and "sample_index" in df.columns:
        side_col = _first_column(df, ("side", "foot", "sensor", "limb"))
        mask = df["event"].astype(str).str.lower().str.contains("heel", na=False)
        if side_col:
            mask &= df[side_col].astype(str).str.lower().isin(side_aliases)
        idx = df.loc[mask, "sample_index"].astype(int).tolist()

    elif side == "left" and "heel_strike_left" in df.columns:
        idx = _indices_from_binary_column(df["heel_strike_left"].values)
    elif side == "right" and "heel_strike_right" in df.columns:
        idx = _indices_from_binary_column(df["heel_strike_right"].values)

    for col in df.columns:
        if "heel" in col and side[:1] in col:
            idx.extend(_indices_from_binary_column(df[col].values))

    if not idx and {"toe_off", "heel_strike"}.issubset(set(df.columns)):
        # Wide format with separate TO/HS index columns per side prefix.
        prefix = "left" if side == "left" else "right"
        hs_cols = [c for c in df.columns if "heel" in c and prefix in c]
        for col in hs_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
            idx.extend(vals.tolist())

    idx = sorted(set(int(i) for i in idx if i >= 0))
    if n_samples is not None:
        idx = [i for i in idx if i < n_samples]
    return np.asarray(idx, dtype=int)


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _indices_from_binary_column(values: np.ndarray) -> list[int]:
    arr = np.asarray(values)
    if arr.dtype == bool or set(np.unique(arr[~np.isnan(arr)])).issubset({0, 1, 0.0, 1.0}):
        return np.where(arr.astype(int) == 1)[0].tolist()
    numeric = pd.to_numeric(arr, errors="coerce").dropna()
    return numeric.astype(int).tolist()


def match_heel_strikes(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_samples: int,
) -> dict[str, float | int]:
    """
    Greedy one-to-one matching within ±tolerance_samples (inclusive).

    Returns TP/FP/FN, precision, recall, F1, mean absolute timing error (s).
    """
    detected = np.asarray(detected, dtype=int)
    ground_truth = np.asarray(ground_truth, dtype=int)
    tol = max(0, int(tolerance_samples))

    matched_det: set[int] = set()
    tp = 0
    errors: list[int] = []

    for gt in ground_truth:
        candidates = [d for d in detected if abs(d - gt) <= tol and d not in matched_det]
        if not candidates:
            continue
        best = min(candidates, key=lambda d: abs(d - gt))
        matched_det.add(best)
        tp += 1
        errors.append(abs(best - gt))

    fp = len(detected) - len(matched_det)
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_abs_error_samples": float(np.mean(errors)) if errors else float("nan"),
        "n_detected": int(len(detected)),
        "n_ground_truth": int(len(ground_truth)),
    }
