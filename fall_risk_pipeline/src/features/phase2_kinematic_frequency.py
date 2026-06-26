"""
Phase 2 kinematic, kinetic & frequency-domain features (Trabassi, Sadeghsalehi, Brognara).

  - Peak angular velocity (deg/s) — HE/LB/LF/RF gyroscopes
  - RMS acceleration — 2 s sliding windows (smoothness / signal energy)
  - Harmonic ratios — LB AP/ML/V even/odd harmonic power
  - Freezing index — PSD ratio 3–8 Hz / 0.5–3 Hz on LB/trunk Z
  - Joint angle ROM (°) — hip (LB), knee (foot−LB differential), ankle (foot) gyro integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.spectral_utils import (
    freezing_index_from_psd,
    harmonic_ratio_even_odd,
    welch_psd,
)

SENSOR_PREFIXES = {
    "head": "head",
    "lower_back": "lb",
    "left_foot": "lf",
    "right_foot": "rf",
}

LB_AXIS_MAP = (
    ("acc_x_grav_free", "acc_x", "ap"),
    ("acc_y_grav_free", "acc_y", "ml"),
    ("acc_z_grav_free", "acc_z", "v"),
)


@dataclass(frozen=True)
class Phase2Config:
    enabled: bool = True
    rms_window_s: float = 2.0
    locomotion_band_hz: tuple[float, float] = (0.5, 3.0)
    freezing_band_hz: tuple[float, float] = (3.0, 8.0)
    fi_sample_window_s: float = 2.0
    joint_gyro_axis: str = "gyr_y"
    gyro_in_degrees: bool = False
    lp_cut_hz: float = 20.0
    harmonic_max_order: int = 6

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Phase2Config:
        feat = (config.get("features") or {}).get("phase2_kinematic_frequency") or {}
        pp = config.get("preprocessing") or {}
        fi = feat.get("freezing_index") or {}
        loco = fi.get("locomotion_band_hz", [0.5, 3.0])
        freeze = fi.get("freezing_band_hz", [3.0, 8.0])
        return cls(
            enabled=bool(feat.get("enabled", True)),
            rms_window_s=float(feat.get("rms_window_s", 2.0)),
            locomotion_band_hz=(float(loco[0]), float(loco[1])),
            freezing_band_hz=(float(freeze[0]), float(freeze[1])),
            fi_sample_window_s=float(fi.get("sample_window_s", 2.0)),
            joint_gyro_axis=str(feat.get("joint_gyro_axis", "gyr_y")),
            gyro_in_degrees=bool(pp.get("gyro_in_degrees", False)),
            lp_cut_hz=float(pp.get("lowpass_cutoff_hz", 20.0)),
            harmonic_max_order=int(feat.get("harmonic_max_order", 6)),
        )


def _gyro_deg_s(df: pd.DataFrame, *, gyro_in_degrees: bool) -> np.ndarray | None:
    if "gyr_resultant" in df.columns:
        gyr = df["gyr_resultant"].to_numpy(dtype=float)
    elif all(c in df.columns for c in ("gyr_x", "gyr_y", "gyr_z")):
        g = df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=float)
        gyr = np.linalg.norm(g, axis=1)
    else:
        return None
    if gyro_in_degrees:
        return gyr
    return np.rad2deg(gyr)


def _acc_for_rms(df: pd.DataFrame) -> np.ndarray | None:
    if "acc_resultant" in df.columns:
        return df["acc_resultant"].to_numpy(dtype=float)
    cols = [c for c in ("acc_x_grav_free", "acc_y_grav_free", "acc_z_grav_free") if c in df.columns]
    if len(cols) == 3:
        return np.linalg.norm(df[cols].to_numpy(dtype=float), axis=1)
    cols = [c for c in ("acc_x", "acc_y", "acc_z") if c in df.columns]
    if len(cols) == 3:
        return np.linalg.norm(df[cols].to_numpy(dtype=float), axis=1)
    return None


def _rolling_rms_stats(sig: np.ndarray, fs: float, window_s: float) -> tuple[float, float]:
    """Mean and std of RMS over non-overlapping *window_s* blocks."""
    arr = np.asarray(sig, dtype=float)
    win = max(1, int(window_s * fs))
    if arr.size < win:
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")
        return rms, float("nan")
    n_win = arr.size // win
    trimmed = arr[: n_win * win].reshape(n_win, win)
    rms_vals = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return float(np.mean(rms_vals)), float(np.std(rms_vals))


def _pick_acc_column(df: pd.DataFrame, grav_free: str, raw: str) -> str | None:
    if grav_free in df.columns:
        return grav_free
    if raw in df.columns:
        return raw
    return None


def _axis_harmonic_ratio(
    df: pd.DataFrame,
    grav_free: str,
    raw: str,
    fs: float,
    lp_cut_hz: float,
    *,
    max_order: int,
) -> float | None:
    col = _pick_acc_column(df, grav_free, raw)
    if col is None:
        return None
    freqs, pxx = welch_psd(df[col].to_numpy(dtype=float), fs)
    if len(pxx) == 0 or float(pxx.sum()) < 1e-12:
        return None
    dom = float(freqs[int(np.argmax(pxx))])
    return harmonic_ratio_even_odd(
        freqs, pxx, dom, lp_cut_hz, max_harmonic_order=max_order
    )


def _event_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=int)
    return np.where(df[col].values == 1)[0]


def _gyr_axis_rad_s(df: pd.DataFrame, axis: str, *, gyro_in_degrees: bool) -> np.ndarray | None:
    if axis not in df.columns:
        return None
    g = df[axis].to_numpy(dtype=float)
    if gyro_in_degrees:
        return np.deg2rad(g)
    return g


def _integrate_angle_rom_deg(gyr_rad_s: np.ndarray, fs: float) -> float:
    """Peak-to-peak integrated angle (°) with endpoint drift correction."""
    seg = np.asarray(gyr_rad_s, dtype=float)
    if seg.size < 4:
        return float("nan")
    dt = 1.0 / float(fs)
    angle = np.cumsum(seg) * dt
    t_norm = np.linspace(0.0, 1.0, len(angle))
    angle_corr = angle - angle[-1] * t_norm
    return float(np.ptp(np.rad2deg(angle_corr)))


def _swing_gyro_segments(
    df: pd.DataFrame,
    side: str,
    gyr_axis: str,
    *,
    gyro_in_degrees: bool,
) -> list[np.ndarray]:
    hs = _event_indices(df, f"heel_strike_{side}")
    to = _event_indices(df, f"toe_off_{side}")
    gyr = _gyr_axis_rad_s(df, gyr_axis, gyro_in_degrees=gyro_in_degrees)
    if gyr is None or len(hs) < 2:
        return []
    segments: list[np.ndarray] = []
    for h in hs[:-1]:
        toe_offs = to[to > h]
        if len(toe_offs) == 0:
            continue
        t0 = int(toe_offs[0])
        next_hs = hs[hs > t0]
        if len(next_hs) == 0:
            continue
        segments.append(gyr[t0 : int(next_hs[0])])
    return segments


def _mean_swing_rom_deg(segments: list[np.ndarray], fs: float) -> float | None:
    roms = [_integrate_angle_rom_deg(s, fs) for s in segments if len(s) >= 4]
    roms = [r for r in roms if np.isfinite(r)]
    return float(np.mean(roms)) if roms else None


def _sensor_kinetic_features(
    df: pd.DataFrame,
    prefix: str,
    fs: float,
    cfg: Phase2Config,
) -> dict[str, float]:
    out: dict[str, float] = {}
    gyr_deg = _gyro_deg_s(df, gyro_in_degrees=cfg.gyro_in_degrees)
    if gyr_deg is not None and gyr_deg.size:
        out[f"{prefix}_peak_gyro_deg_s"] = float(np.max(gyr_deg))

    acc = _acc_for_rms(df)
    if acc is not None and acc.size:
        rms_mean, rms_std = _rolling_rms_stats(acc, fs, cfg.rms_window_s)
        out[f"{prefix}_rms_acc_2s_mean"] = rms_mean
        if np.isfinite(rms_std):
            out[f"{prefix}_rms_acc_2s_std"] = rms_std
    return out


def _lb_frequency_features(df: pd.DataFrame, fs: float, cfg: Phase2Config) -> dict[str, float]:
    out: dict[str, float] = {}
    for grav_free, raw, label in LB_AXIS_MAP:
        hr = _axis_harmonic_ratio(
            df, grav_free, raw, fs, cfg.lp_cut_hz, max_order=cfg.harmonic_max_order
        )
        if hr is not None:
            out[f"lb_harmonic_ratio_{label}"] = hr

    z_col = _pick_acc_column(df, "acc_z_grav_free", "acc_z")
    if z_col is not None:
        freqs, pxx = welch_psd(df[z_col].to_numpy(dtype=float), fs)
        fi = freezing_index_from_psd(
            freqs,
            pxx,
            locomotion_band_hz=cfg.locomotion_band_hz,
            freezing_band_hz=cfg.freezing_band_hz,
        )
        if np.isfinite(fi):
            out["lb_freezing_index"] = fi
    return out


def _swing_segment_bounds(hs: np.ndarray, to: np.ndarray) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for h in hs[:-1]:
        toe_offs = to[to > h]
        if len(toe_offs) == 0:
            continue
        t0 = int(toe_offs[0])
        next_hs = hs[hs > t0]
        if len(next_hs) == 0:
            continue
        bounds.append((t0, int(next_hs[0])))
    return bounds


def _joint_angle_features(
    lb: pd.DataFrame | None,
    lf: pd.DataFrame | None,
    rf: pd.DataFrame | None,
    hd: pd.DataFrame | None,
    fs: float,
    cfg: Phase2Config,
) -> dict[str, float]:
    out: dict[str, float] = {}

    # Hip ROM: LB sagittal gyro integrated per stride (HS → HS).
    if lb is not None:
        gyr_lb = _gyr_axis_rad_s(lb, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees)
        if gyr_lb is not None:
            hs_l = _event_indices(lf, "heel_strike_left") if lf is not None else np.array([])
            hs_r = _event_indices(rf, "heel_strike_right") if rf is not None else np.array([])
            hs_all = np.sort(np.unique(np.concatenate([hs_l, hs_r]))) if len(hs_l) or len(hs_r) else np.array([])
            hip_roms: list[float] = []
            for i in range(len(hs_all) - 1):
                seg = gyr_lb[int(hs_all[i]) : int(hs_all[i + 1])]
                rom = _integrate_angle_rom_deg(seg, fs)
                if np.isfinite(rom):
                    hip_roms.append(rom)
            if hip_roms:
                out["hip_angle_rom_deg"] = float(np.mean(hip_roms))

    # Ankle ROM: foot gyro during swing (bilateral mean).
    ankle_roms: list[float] = []
    for foot, side in ((lf, "left"), (rf, "right")):
        if foot is None:
            continue
        for seg in _swing_gyro_segments(
            foot, side, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees
        ):
            rom = _integrate_angle_rom_deg(seg, fs)
            if np.isfinite(rom):
                ankle_roms.append(rom)
    if ankle_roms:
        out["ankle_angle_rom_deg"] = float(np.mean(ankle_roms))

    # Knee ROM proxy: (foot − LB) gyro during swing.
    knee_roms: list[float] = []
    if lb is not None:
        gyr_lb = _gyr_axis_rad_s(lb, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees)
        for foot, side in ((lf, "left"), (rf, "right")):
            if foot is None or gyr_lb is None:
                continue
            gyr_f = _gyr_axis_rad_s(foot, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees)
            if gyr_f is None:
                continue
            hs = _event_indices(foot, f"heel_strike_{side}")
            to = _event_indices(foot, f"toe_off_{side}")
            for t0, t1 in _swing_segment_bounds(hs, to):
                n = min(t1, len(gyr_f), len(gyr_lb))
                if n <= t0 + 4:
                    continue
                diff = gyr_f[t0:n] - gyr_lb[t0:n]
                rom = _integrate_angle_rom_deg(diff, fs)
                if np.isfinite(rom):
                    knee_roms.append(rom)
    if knee_roms:
        out["knee_angle_rom_deg"] = float(np.mean(knee_roms))

    # Head-assisted hip flexion proxy when feet absent (DAPHNET LB-only: skip ankle/knee).
    if hd is not None and lb is not None and "hip_angle_rom_deg" not in out:
        gyr_h = _gyr_axis_rad_s(hd, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees)
        gyr_lb = _gyr_axis_rad_s(lb, cfg.joint_gyro_axis, gyro_in_degrees=cfg.gyro_in_degrees)
        if gyr_h is not None and gyr_lb is not None:
            n = min(len(gyr_h), len(gyr_lb))
            diff = gyr_h[:n] - gyr_lb[:n]
            rom = _integrate_angle_rom_deg(diff, fs)
            if np.isfinite(rom):
                out["hip_angle_rom_deg"] = rom

    return out


def extract_phase2_kinematic_frequency_features(
    signals: dict[str, pd.DataFrame],
    *,
    fs: float,
    config: dict[str, Any] | Phase2Config | None = None,
) -> dict[str, float]:
    """Trial-level Phase 2 features from preprocessed sensor dict."""
    cfg = (
        config
        if isinstance(config, Phase2Config)
        else Phase2Config.from_config(config or {})
    )
    if not cfg.enabled:
        return {}

    feats: dict[str, float] = {}
    for pos, prefix in SENSOR_PREFIXES.items():
        df = signals.get(pos)
        if df is None:
            continue
        feats.update(_sensor_kinetic_features(df, prefix, fs, cfg))

    lb = signals.get("lower_back")
    if lb is not None:
        feats.update(_lb_frequency_features(lb, fs, cfg))

    feats.update(
        _joint_angle_features(
            lb,
            signals.get("left_foot"),
            signals.get("right_foot"),
            signals.get("head"),
            fs,
            cfg,
        )
    )
    return feats
