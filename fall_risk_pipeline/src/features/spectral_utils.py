"""
Shared spectral helpers for feature extraction (Phase 2 + legacy spectral group).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def spectral_centroid_hz(freqs: np.ndarray, pxx: np.ndarray) -> float:
    """First moment of the PSD: sum(f * P) / sum(P) in Hz."""
    pxx = np.asarray(pxx, dtype=float)
    total = float(pxx.sum())
    if total < 1e-12:
        return float("nan")
    return float(np.sum(np.asarray(freqs, dtype=float) * pxx) / total)


def psd_band_power(
    freqs: np.ndarray,
    pxx: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> float:
    """Integrate PSD between f_lo and f_hi (Hz)."""
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not mask.any():
        return 0.0
    fm, pm = freqs[mask], pxx[mask]
    if len(fm) > 1:
        integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        if integrator is None:
            return float(np.sum(pm))
        return float(integrator(pm, fm))
    return float(pm[0])


def harmonic_ratio_even_odd(
    freqs: np.ndarray,
    pxx: np.ndarray,
    dominant_freq: float,
    lp_cut_hz: float,
    *,
    max_harmonic_order: int = 6,
) -> float | None:
    """Even/odd harmonic power ratio using only in-band harmonics."""
    if dominant_freq <= 0:
        return None

    max_harmonic_hz = lp_cut_hz * 0.8
    even_sum = 0.0
    odd_sum = 0.0
    for k in range(1, max_harmonic_order + 1):
        harmonic_hz = dominant_freq * k
        if harmonic_hz >= max_harmonic_hz:
            continue
        power = float(pxx[np.argmin(np.abs(freqs - harmonic_hz))])
        if k % 2 == 1:
            odd_sum += power
        else:
            even_sum += power

    if even_sum <= 0.0 and odd_sum <= 0.0:
        return None
    return float(even_sum / (odd_sum + 1e-10))


def welch_psd(
    sig: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sig, dtype=float)
    if arr.size < 8:
        return np.array([]), np.array([])
    seg = nperseg or min(256, max(4, arr.size // 2))
    return welch(arr, fs=fs, nperseg=seg)


def freezing_index_from_psd(
    freqs: np.ndarray,
    pxx: np.ndarray,
    *,
    locomotion_band_hz: tuple[float, float] = (0.5, 3.0),
    freezing_band_hz: tuple[float, float] = (3.0, 8.0),
) -> float:
    """
    Freezing index = power(freezing_band) / power(locomotion_band).

  Salchow-Hömmen / DAPHNET convention on trunk vertical acceleration.
    """
    if len(pxx) == 0:
        return float("nan")
    p_loco = psd_band_power(freqs, pxx, *locomotion_band_hz)
    p_freeze = psd_band_power(freqs, pxx, *freezing_band_hz)
    if p_loco <= 1e-12:
        return float("nan")
    return float(p_freeze / p_loco)


def sample_freezing_index_series(
    sig: np.ndarray,
    fs: float,
    *,
    window_s: float = 2.0,
    hop_s: float = 0.1,
    locomotion_band_hz: tuple[float, float] = (0.5, 3.0),
    freezing_band_hz: tuple[float, float] = (3.0, 8.0),
) -> np.ndarray:
    """Per-sample freezing index via short-time Welch (forward-filled within each hop)."""
    arr = np.asarray(sig, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    win = max(8, int(window_s * fs))
    hop = max(1, int(hop_s * fs))
    if n < win:
        freqs, pxx = welch_psd(arr, fs)
        fi = freezing_index_from_psd(
            freqs, pxx,
            locomotion_band_hz=locomotion_band_hz,
            freezing_band_hz=freezing_band_hz,
        )
        out[:] = fi
        return out

    for start in range(0, n - win + 1, hop):
        seg = arr[start : start + win]
        freqs, pxx = welch_psd(seg, fs)
        fi = freezing_index_from_psd(
            freqs, pxx,
            locomotion_band_hz=locomotion_band_hz,
            freezing_band_hz=freezing_band_hz,
        )
        out[start : min(n, start + hop)] = fi
    # Fill tail with last computed value
    last = np.nan
    for i in range(n):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    return out
