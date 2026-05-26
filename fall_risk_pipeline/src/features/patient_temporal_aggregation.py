"""
Patient-level aggregation across within-session trials.

Beyond mean and std, captures intra-session variability (range) and
systematic change across ordered trials (linear trend / slope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_patient_aggregation_config(features_cfg: dict) -> dict:
    """Resolve aggregation flags from config (supports legacy string form)."""
    raw = features_cfg.get("patient_aggregation", {})
    if isinstance(raw, str):
        legacy = raw.lower()
        return {
            "include_mean": "mean" in legacy,
            "include_std": "std" in legacy,
            "include_range": False,
            "include_trend": False,
            "trial_order": ["session", "trial_id"],
            "min_trials_for_trend": 2,
        }
    if not isinstance(raw, dict):
        raw = {}
    return {
        "include_mean": bool(raw.get("include_mean", True)),
        "include_std": bool(raw.get("include_std", True)),
        "include_range": bool(raw.get("include_range", True)),
        "include_trend": bool(raw.get("include_trend", True)),
        "trial_order": list(raw.get("trial_order", ["session", "trial_id"])),
        "min_trials_for_trend": int(raw.get("min_trials_for_trend", 2)),
    }


def order_trial_group(grp: pd.DataFrame, order_cols: list[str]) -> pd.DataFrame:
    """Sort trials for temporal trend (session, then trial_id by default)."""
    cols = [c for c in order_cols if c in grp.columns]
    if not cols:
        return grp.sort_values("trial_id") if "trial_id" in grp.columns else grp
    return grp.sort_values(cols, kind="mergesort")


def trial_feature_range(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    if len(v) == 1:
        return 0.0
    return float(np.max(v) - np.min(v))


def trial_feature_trend_slope(
    values: np.ndarray,
    *,
    min_trials: int = 2,
) -> float:
    """
    OLS slope of feature vs trial order index (0 .. n-1).

    Positive slope → feature increases over successive trials in the session.
    """
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    n = len(y)
    if n < min_trials:
        return float("nan")
    x = np.arange(n, dtype=float)
    if np.std(y) < 1e-12:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def aggregate_trial_values(
    values: np.ndarray,
    agg_cfg: dict,
) -> dict[str, float]:
    """Compute mean/std/range/trend for one feature across ordered trials."""
    v = np.asarray(values, dtype=float)
    v_finite = v[np.isfinite(v)]
    out: dict[str, float] = {}

    if agg_cfg.get("include_mean", True):
        out["mean"] = float(np.mean(v_finite)) if len(v_finite) else float("nan")
    if agg_cfg.get("include_std", True):
        out["std"] = float(np.std(v_finite)) if len(v_finite) else float("nan")
    if agg_cfg.get("include_range", True):
        out["range"] = trial_feature_range(v_finite)
    if agg_cfg.get("include_trend", True):
        out["trend"] = trial_feature_trend_slope(
            v,
            min_trials=int(agg_cfg.get("min_trials_for_trend", 2)),
        )
    return out
