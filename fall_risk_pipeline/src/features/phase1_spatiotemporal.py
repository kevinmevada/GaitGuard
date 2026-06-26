"""
Phase 1 spatiotemporal + variability features (Moon 2020, Trabassi 2022, Voisard).

Derived from foot IMU gait events (heel strike / toe-off) on LF/RF:
  - Stride & step duration (s)
  - Stance / swing phase (%)
  - Stride & step length (m) — swing-phase double integration + drift correction
  - Gait velocity (m/s)
  - Coefficient of variation (CV%) over rolling stride windows
  - Symmetry index (SI) — bilateral comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDES = (SIDE_LEFT, SIDE_RIGHT)


@dataclass(frozen=True)
class Phase1Config:
    enabled: bool = True
    rolling_cv_window_strides: int = 5
    enable_spatial_integration: bool = True
    min_strides: int = 3
    integration_axis: str = "acc_resultant"  # acc_x | acc_resultant

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Phase1Config:
        feat = (config.get("features") or {}).get("phase1_spatiotemporal") or {}
        return cls(
            enabled=bool(feat.get("enabled", True)),
            rolling_cv_window_strides=int(feat.get("rolling_cv_window_strides", 5)),
            enable_spatial_integration=bool(feat.get("enable_spatial_integration", True)),
            min_strides=int(feat.get("min_strides", 3)),
            integration_axis=str(feat.get("integration_axis", "acc_resultant")),
        )


def symmetry_index(left: float, right: float) -> float:
    """SI (%) = |L − R| / (0.5 × (L + R)) × 100 (Herzog / Robinson convention)."""
    denom = 0.5 * (left + right)
    if denom <= 1e-12 or not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(abs(left - right) / denom * 100.0)


def coefficient_of_variation_pct(values: np.ndarray) -> float:
    """(SD / Mean) × 100."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(arr) / mean * 100.0)


def rolling_cv_pct(values: np.ndarray, window: int) -> float:
    """Mean of per-window CV% over a rolling stride window."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < window or window < 2:
        return coefficient_of_variation_pct(arr)
    cvs: list[float] = []
    for start in range(0, len(arr) - window + 1):
        chunk = arr[start : start + window]
        cv = coefficient_of_variation_pct(chunk)
        if np.isfinite(cv):
            cvs.append(cv)
    return float(np.mean(cvs)) if cvs else float("nan")


def _event_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=int)
    return np.where(df[col].values == 1)[0]


def _stride_durations_s(hs_idx: np.ndarray, fs: float) -> np.ndarray:
    if len(hs_idx) < 2:
        return np.array([])
    return np.diff(hs_idx).astype(float) / float(fs)


def _step_durations_s(hs_left: np.ndarray, hs_right: np.ndarray, fs: float) -> np.ndarray:
    """Alternating heel strikes (contralateral step times)."""
    if len(hs_left) == 0 or len(hs_right) == 0:
        return np.array([])
    events = [(int(i), SIDE_LEFT) for i in hs_left] + [(int(i), SIDE_RIGHT) for i in hs_right]
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        return np.array([])
    times = np.array([e[0] for e in events], dtype=float) / float(fs)
    return np.diff(times)


def _stance_swing_durations(
    hs_idx: np.ndarray,
    to_idx: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    stance: list[float] = []
    swing: list[float] = []
    for hs in hs_idx[:-1]:
        toe_offs_after = to_idx[to_idx > hs]
        if len(toe_offs_after) == 0:
            continue
        to_sample = int(toe_offs_after[0])
        stance.append((to_sample - hs) / float(fs))
        next_hs = hs_idx[hs_idx > to_sample]
        if len(next_hs) > 0:
            swing.append((int(next_hs[0]) - to_sample) / float(fs))
    return np.asarray(stance, dtype=float), np.asarray(swing, dtype=float)


def _integrate_step_length_m(
    acc: np.ndarray,
    fs: float,
) -> float:
    """
    Double integration with endpoint drift correction (velocity zeroed at segment ends).

    Assumes acc is the forward/swing-phase horizontal component (m/s²).
    """
    seg = np.asarray(acc, dtype=float)
    if seg.size < 4 or not np.isfinite(seg).all():
        return float("nan")
    dt = 1.0 / float(fs)
    vel = np.cumsum(seg) * dt
    t_norm = np.linspace(0.0, 1.0, len(vel))
    vel_corr = vel - vel[-1] * t_norm
    disp = np.cumsum(vel_corr) * dt
    return float(abs(disp[-1]))


def _swing_step_lengths_m(
    df: pd.DataFrame,
    hs_idx: np.ndarray,
    to_idx: np.ndarray,
    fs: float,
    *,
    acc_col: str,
) -> np.ndarray:
    if acc_col not in df.columns or len(hs_idx) < 2:
        return np.array([])
    lengths: list[float] = []
    acc = df[acc_col].to_numpy(dtype=float)
    for hs in hs_idx[:-1]:
        toe_offs_after = to_idx[to_idx > hs]
        if len(toe_offs_after) == 0:
            continue
        to_sample = int(toe_offs_after[0])
        next_hs = hs_idx[hs_idx > to_sample]
        if len(next_hs) == 0:
            continue
        end = int(next_hs[0])
        seg = acc[to_sample:end]
        length = _integrate_step_length_m(seg, fs)
        if np.isfinite(length) and length > 0:
            lengths.append(length)
    return np.asarray(lengths, dtype=float)


def _side_timing_features(
    df: pd.DataFrame,
    side: str,
    fs: float,
    *,
    cfg: Phase1Config,
) -> dict[str, float]:
    hs_idx = _event_indices(df, f"heel_strike_{side}")
    to_idx = _event_indices(df, f"toe_off_{side}")
    out: dict[str, float] = {}

    strides = _stride_durations_s(hs_idx, fs)
    if strides.size >= cfg.min_strides:
        out[f"{side}_stride_duration_s"] = float(np.mean(strides))
        out[f"{side}_stride_duration_std"] = float(np.std(strides))
        out[f"{side}_stride_duration_cv_pct"] = coefficient_of_variation_pct(strides)
        out[f"{side}_stride_duration_rolling_cv_pct"] = rolling_cv_pct(
            strides, cfg.rolling_cv_window_strides
        )
        out[f"{side}_cadence_spm"] = float(60.0 / (out[f"{side}_stride_duration_s"] + 1e-10))

    stance, swing = _stance_swing_durations(hs_idx, to_idx, fs)
    stride_ref = out.get(f"{side}_stride_duration_s", np.nan)
    if stance.size and np.isfinite(stride_ref) and stride_ref > 0:
        out[f"{side}_stance_pct"] = float(np.mean(stance) / stride_ref * 100.0)
    if swing.size and np.isfinite(stride_ref) and stride_ref > 0:
        out[f"{side}_swing_pct"] = float(np.mean(swing) / stride_ref * 100.0)

    if cfg.enable_spatial_integration:
        step_lens = _swing_step_lengths_m(
            df, hs_idx, to_idx, fs, acc_col=cfg.integration_axis
        )
        if step_lens.size >= 2:
            out[f"{side}_step_length_m"] = float(np.mean(step_lens))
            out[f"{side}_stride_length_m"] = float(np.mean(step_lens))
            step_dur = swing if swing.size else strides
            if step_dur.size >= 2:
                mean_dur = float(np.mean(step_dur))
                if mean_dur > 1e-6:
                    out[f"{side}_gait_velocity_m_s"] = float(
                        out[f"{side}_step_length_m"] / mean_dur
                    )

    out[f"{side}_step_count"] = float(len(hs_idx))
    return out


def extract_phase1_spatiotemporal_features(
    lf: pd.DataFrame,
    rf: pd.DataFrame,
    *,
    fs: float,
    config: dict[str, Any] | Phase1Config | None = None,
) -> dict[str, float]:
    """Compute Phase 1 trial-level spatiotemporal + variability features."""
    cfg = (
        config
        if isinstance(config, Phase1Config)
        else Phase1Config.from_config(config or {})
    )
    if not cfg.enabled:
        return {}

    feats: dict[str, float] = {}
    per_side: dict[str, dict[str, float]] = {}
    for df, side in ((lf, SIDE_LEFT), (rf, SIDE_RIGHT)):
        per_side[side] = _side_timing_features(df, side, fs, cfg=cfg)
        feats.update(per_side[side])

    def _bilateral_mean(suffix: str) -> float | None:
        vals = [
            per_side[s].get(f"{s}_{suffix}")
            for s in SIDES
            if f"{s}_{suffix}" in per_side[s]
        ]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    for suffix in (
        "stride_duration_s",
        "stride_duration_std",
        "stride_duration_cv_pct",
        "stride_duration_rolling_cv_pct",
        "stance_pct",
        "swing_pct",
        "step_length_m",
        "stride_length_m",
        "gait_velocity_m_s",
        "cadence_spm",
    ):
        m = _bilateral_mean(suffix)
        if m is not None:
            feats[suffix] = m

    hs_l = _event_indices(lf, "heel_strike_left")
    hs_r = _event_indices(rf, "heel_strike_right")
    steps = _step_durations_s(hs_l, hs_r, fs)
    if steps.size >= cfg.min_strides:
        feats["step_duration_s"] = float(np.mean(steps))
        feats["step_duration_cv_pct"] = coefficient_of_variation_pct(steps)
        feats["step_duration_rolling_cv_pct"] = rolling_cv_pct(
            steps, cfg.rolling_cv_window_strides
        )

    step_count = sum(int(per_side[s].get(f"{s}_step_count", 0)) for s in SIDES)
    if step_count:
        feats["step_count"] = float(step_count)

    si_pairs = (
        ("si_stride_duration", "stride_duration_s"),
        ("si_stance_pct", "stance_pct"),
        ("si_swing_pct", "swing_pct"),
        ("si_step_length", "step_length_m"),
        ("si_gait_velocity", "gait_velocity_m_s"),
        ("si_stride_duration_cv_pct", "stride_duration_cv_pct"),
    )
    for si_name, suffix in si_pairs:
        lv = per_side[SIDE_LEFT].get(f"{SIDE_LEFT}_{suffix}")
        rv = per_side[SIDE_RIGHT].get(f"{SIDE_RIGHT}_{suffix}")
        if lv is not None and rv is not None and np.isfinite(lv) and np.isfinite(rv):
            feats[si_name] = symmetry_index(lv, rv)

    _apply_legacy_aliases(feats, per_side)
    return feats


def _apply_legacy_aliases(feats: dict[str, float], per_side: dict[str, dict[str, float]]) -> None:
    """Map Phase 1 names to legacy feature_extractor keys for backward compatibility."""
    if "stride_duration_s" in feats:
        feats["stride_time_mean"] = feats["stride_duration_s"]
    if "stride_duration_std" in feats:
        feats["stride_time_std"] = feats["stride_duration_std"]
    if "stride_duration_cv_pct" in feats:
        feats["stride_time_cv"] = feats["stride_duration_cv_pct"] / 100.0
    if "stance_pct" in feats:
        feats["stance_phase_ratio"] = feats["stance_pct"] / 100.0
    if "swing_pct" in feats:
        feats["swing_phase_ratio"] = feats["swing_pct"] / 100.0
    if "cadence_spm" in feats:
        feats["cadence"] = feats["cadence_spm"]
    cads = [feats[f"{s}_cadence"] for s in SIDES if f"{s}_cadence" in feats]
    if cads:
        feats["cadence_mean"] = float(np.mean(cads))

    for side in SIDES:
        sd = per_side[side].get(f"{side}_stride_duration_s")
        if sd is not None:
            feats[f"{side}_stride_time_mean"] = sd
        sstd = per_side[side].get(f"{side}_stride_duration_std")
        if sstd is not None:
            feats[f"{side}_stride_time_std"] = sstd
        scv = per_side[side].get(f"{side}_stride_duration_cv_pct")
        if scv is not None:
            feats[f"{side}_stride_time_cv"] = scv / 100.0
        if f"{side}_cadence_spm" in per_side[side]:
            feats[f"{side}_cadence"] = per_side[side][f"{side}_cadence_spm"]

    if "si_stride_duration" in feats:
        feats["stride_time_mean_asymmetry"] = feats["si_stride_duration"] / 100.0
    if "si_stride_duration_cv_pct" in feats:
        feats["stride_time_std_asymmetry"] = feats["si_stride_duration_cv_pct"] / 100.0

    if "stance_pct" in feats and "swing_pct" in feats:
        ds = feats["stance_phase_ratio"] + feats["swing_phase_ratio"] - 1.0
        feats["double_support_ratio"] = max(0.0, min(1.0, ds))
