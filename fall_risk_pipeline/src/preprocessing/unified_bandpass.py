"""
Stage C — unified accelerometer bandpass (Voisard + DAPHNET merge).

After both corpora are at 100 Hz, apply the **same** zero-phase Butterworth
bandpass to all accelerometer channels:

  4th-order, 0.5–20 Hz, ``filtfilt`` (no group-delay peak shift on stride events).

Any asymmetry between datasets at this step invalidates cross-dataset comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ACC_AXIS_COLUMNS = ("acc_x", "acc_y", "acc_z")

DEFAULT_BANDPASS_ORDER = 4
DEFAULT_BANDPASS_LOW_HZ = 0.5
DEFAULT_BANDPASS_HIGH_HZ = 20.0
MIN_SAMPLES_BANDPASS = 12  # > 3 * order for filtfilt padlen


@dataclass(frozen=True)
class UnifiedBandpassConfig:
    enabled: bool = True
    low_hz: float = DEFAULT_BANDPASS_LOW_HZ
    high_hz: float = DEFAULT_BANDPASS_HIGH_HZ
    order: int = DEFAULT_BANDPASS_ORDER
    fs_hz: float = 100.0

    @classmethod
    def from_pipeline_config(cls, config: dict) -> UnifiedBandpassConfig:
        pp = config.get("preprocessing") or {}
        ucfg = pp.get("unified_acc_bandpass") or {}
        fs = float((config.get("dataset") or {}).get("sampling_rate", 100.0))
        return cls(
            enabled=bool(ucfg.get("enabled", True)),
            low_hz=float(ucfg.get("low_hz", DEFAULT_BANDPASS_LOW_HZ)),
            high_hz=float(ucfg.get("high_hz", DEFAULT_BANDPASS_HIGH_HZ)),
            order=int(ucfg.get("order", DEFAULT_BANDPASS_ORDER)),
            fs_hz=fs,
        )


def bandpass_coefficients(
    *,
    fs_hz: float,
    low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    order: int = DEFAULT_BANDPASS_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass coefficients (0.5–20 Hz @ 100 Hz by default)."""
    if low_hz <= 0 or high_hz <= low_hz or high_hz >= fs_hz / 2:
        raise ValueError(
            f"Invalid bandpass edges: low={low_hz}, high={high_hz}, fs={fs_hz}"
        )
    try:
        b, a = butter(order, [low_hz, high_hz], btype="band", fs=fs_hz)
    except TypeError:
        nyq = fs_hz / 2.0
        b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return b, a


def filtfilt_bandpass(
    signal: np.ndarray,
    *,
    fs_hz: float,
    low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    order: int = DEFAULT_BANDPASS_ORDER,
    axis: int = 0,
) -> np.ndarray:
    """Zero-phase bandpass along ``axis`` (default: time / rows)."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return arr
    n_time = arr.shape[axis]
    if n_time < MIN_SAMPLES_BANDPASS:
        return arr
    b, a = bandpass_coefficients(
        fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, order=order
    )
    return filtfilt(b, a, arr, axis=axis)


def apply_unified_acc_bandpass(
    df: pd.DataFrame,
    cfg: UnifiedBandpassConfig,
) -> pd.DataFrame:
    """
    Filter ``acc_x``, ``acc_y``, ``acc_z`` identically (Stage C merge filter).
    """
    if not cfg.enabled:
        return df
    acc_cols = [c for c in ACC_AXIS_COLUMNS if c in df.columns]
    if not acc_cols:
        return df
    if len(df) < MIN_SAMPLES_BANDPASS:
        return df

    out = df.copy()
    mat = out[acc_cols].to_numpy(dtype=np.float64)
    filtered = filtfilt_bandpass(
        mat,
        fs_hz=cfg.fs_hz,
        low_hz=cfg.low_hz,
        high_hz=cfg.high_hz,
        order=cfg.order,
        axis=0,
    )
    out[acc_cols] = filtered
    return out


def lowpass_gyro_columns(
    df: pd.DataFrame,
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
) -> pd.DataFrame:
    """Low-pass gyro axes to the unified upper band (Voisard only)."""
    gyr_cols = [c for c in df.columns if c.startswith("gyr_")]
    if not gyr_cols or len(df) < order * 3:
        return df
    out = df.copy()
    try:
        b, a = butter(order, cutoff_hz, btype="low", fs=fs_hz)
    except TypeError:
        b, a = butter(order, cutoff_hz / (fs_hz / 2.0), btype="low")
    for col in gyr_cols:
        out[col] = filtfilt(b, a, out[col].to_numpy(dtype=np.float64))
    return out
