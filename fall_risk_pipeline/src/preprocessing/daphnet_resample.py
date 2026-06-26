"""
Resample DAPHNET accelerometry 64 Hz → 100 Hz for pipeline parity with Voisard.

Uses ``scipy.signal.resample_poly(up=25, down=16)`` exactly (25/16 = 100/64) to
avoid distortion in the 3–8 Hz gait band (freezing-index numerator).

Applied per-axis on ankle, thigh, and trunk independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import resample_poly, welch

DAPHNET_SOURCE_FS_HZ = 64.0
PIPELINE_TARGET_FS_HZ = 100.0
RESAMPLE_UP = 25
RESAMPLE_DOWN = 16
PSD_BAND_HZ = (3.0, 8.0)
PSD_MAX_PEAK_SHIFT_HZ = 0.5

ACC_COLUMNS = ("acc_x", "acc_y", "acc_z")


class DaphnetResampleError(RuntimeError):
    """Raised when PSD verification fails after resampling."""


@dataclass(frozen=True)
class PsdVerificationResult:
    subject_id: str
    peak_hz_before: float
    peak_hz_after: float
    peak_shift_hz: float
    passed: bool
    figure_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "peak_hz_before": self.peak_hz_before,
            "peak_hz_after": self.peak_hz_after,
            "peak_shift_hz": self.peak_shift_hz,
            "passed": self.passed,
            "figure_path": self.figure_path or "",
        }


def resample_axis(
    values: np.ndarray,
    *,
    up: int = RESAMPLE_UP,
    down: int = RESAMPLE_DOWN,
) -> np.ndarray:
    """Polyphase resample one axis (64 → 100 Hz with default up/down)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return resample_poly(arr, up, down).astype(np.float32)


def resample_sensor_dataframe(
    df: pd.DataFrame,
    *,
    up: int = RESAMPLE_UP,
    down: int = RESAMPLE_DOWN,
    target_fs_hz: float = PIPELINE_TARGET_FS_HZ,
) -> pd.DataFrame:
    """Resample ``acc_x/y/z`` independently; rebuild ``time`` at ``target_fs_hz``."""
    if df.empty:
        return df.copy()

    resampled = {
        col: resample_axis(df[col].to_numpy(), up=up, down=down)
        for col in ACC_COLUMNS
        if col in df.columns
    }
    n_out = len(next(iter(resampled.values())))
    out = pd.DataFrame(resampled)
    out.insert(0, "time", np.arange(n_out, dtype=np.float64) / float(target_fs_hz))
    return out


def resample_daphnet_signals(
    signals: dict[str, pd.DataFrame],
    *,
    up: int = RESAMPLE_UP,
    down: int = RESAMPLE_DOWN,
    target_fs_hz: float = PIPELINE_TARGET_FS_HZ,
) -> dict[str, pd.DataFrame]:
    """Resample all DAPHNET sensors (ankle, thigh, trunk) per-axis."""
    return {
        sensor: resample_sensor_dataframe(
            frame,
            up=up,
            down=down,
            target_fs_hz=target_fs_hz,
        )
        for sensor, frame in signals.items()
    }


def band_peak_frequency(
    signal: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = PSD_BAND_HZ,
) -> float:
    """Dominant Welch PSD peak frequency inside ``band_hz``."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size < 16 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return float("nan")

    nperseg = min(256, max(16, len(arr) // 4))
    freqs, psd = welch(arr, fs=fs_hz, nperseg=nperseg)
    lo, hi = band_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")

    band_freqs = freqs[mask]
    band_psd = psd[mask]
    return float(band_freqs[int(np.argmax(band_psd))])


def plot_trunk_z_psd_check(
    trunk_z_before: np.ndarray,
    trunk_z_after: np.ndarray,
    *,
    subject_id: str,
    fs_before: float,
    fs_after: float,
    out_dir: Path,
    band_hz: tuple[float, float] = PSD_BAND_HZ,
    dpi: int = 150,
) -> Path:
    """Save before/after trunk Z PSD overlay for supplemental material."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"psd_check_{subject_id}.png"

    def _psd(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        nperseg = min(256, max(16, len(sig) // 4))
        return welch(np.asarray(sig, dtype=np.float64), fs=fs, nperseg=nperseg)

    f_before, p_before = _psd(trunk_z_before, fs_before)
    f_after, p_after = _psd(trunk_z_after, fs_after)

    peak_before = band_peak_frequency(trunk_z_before, fs_before, band_hz)
    peak_after = band_peak_frequency(trunk_z_after, fs_after, band_hz)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(f_before, p_before, label=f"64 Hz (peak {peak_before:.2f} Hz)", alpha=0.85)
    ax.semilogy(f_after, p_after, label=f"100 Hz (peak {peak_after:.2f} Hz)", alpha=0.85)
    ax.axvspan(band_hz[0], band_hz[1], color="gold", alpha=0.15, label="3–8 Hz band")
    ax.set_xlim(0, 15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"DAPHNET trunk Z PSD — {subject_id} (resample_poly 25/16)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def verify_trunk_z_resample(
    trunk_z_before: np.ndarray,
    trunk_z_after: np.ndarray,
    *,
    subject_id: str,
    fs_before: float = DAPHNET_SOURCE_FS_HZ,
    fs_after: float = PIPELINE_TARGET_FS_HZ,
    band_hz: tuple[float, float] = PSD_BAND_HZ,
    max_peak_shift_hz: float = PSD_MAX_PEAK_SHIFT_HZ,
    figure_dir: Path | None = None,
    write_figure: bool = True,
) -> PsdVerificationResult:
    """
    Confirm 3–8 Hz trunk-Z peak is preserved within ``max_peak_shift_hz``.

    Raises ``DaphnetResampleError`` when verification fails.
    """
    peak_before = band_peak_frequency(trunk_z_before, fs_before, band_hz)
    peak_after = band_peak_frequency(trunk_z_after, fs_after, band_hz)
    shift = abs(peak_after - peak_before) if np.isfinite(peak_before) and np.isfinite(peak_after) else float("inf")
    passed = bool(np.isfinite(shift) and shift <= max_peak_shift_hz)

    figure_path: str | None = None
    if write_figure and figure_dir is not None:
        fig_path = plot_trunk_z_psd_check(
            trunk_z_before,
            trunk_z_after,
            subject_id=subject_id,
            fs_before=fs_before,
            fs_after=fs_after,
            out_dir=figure_dir,
            band_hz=band_hz,
        )
        figure_path = str(fig_path)

    result = PsdVerificationResult(
        subject_id=subject_id,
        peak_hz_before=peak_before,
        peak_hz_after=peak_after,
        peak_shift_hz=float(shift) if np.isfinite(shift) else float("nan"),
        passed=passed,
        figure_path=figure_path,
    )

    if not passed:
        raise DaphnetResampleError(
            f"DAPHNET PSD check failed for {subject_id}: "
            f"3–8 Hz peak shifted {shift:.3f} Hz "
            f"(before={peak_before:.3f}, after={peak_after:.3f}); "
            f"limit ±{max_peak_shift_hz} Hz. "
            "resample_poly parameters may be wrong — stop and debug."
        )

    logger.info(
        "DAPHNET PSD OK {}: peak {:.3f}→{:.3f} Hz (Δ={:.3f})",
        subject_id,
        peak_before,
        peak_after,
        shift,
    )
    return result


def run_psd_verification_batch(
    subjects: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    figure_dir: Path,
    fs_before: float = DAPHNET_SOURCE_FS_HZ,
    fs_after: float = PIPELINE_TARGET_FS_HZ,
    band_hz: tuple[float, float] = PSD_BAND_HZ,
    max_peak_shift_hz: float = PSD_MAX_PEAK_SHIFT_HZ,
    min_subjects: int = 2,
) -> list[PsdVerificationResult]:
    """
    PSD-verify trunk Z for every subject; save plots for the first ``min_subjects``.
    """
    if len(subjects) < min_subjects:
        raise DaphnetResampleError(
            f"DAPHNET PSD verification requires ≥{min_subjects} subjects; got {len(subjects)}"
        )

    results: list[PsdVerificationResult] = []
    for idx, (subject_id, before, after) in enumerate(subjects):
        results.append(
            verify_trunk_z_resample(
                before,
                after,
                subject_id=subject_id,
                fs_before=fs_before,
                fs_after=fs_after,
                band_hz=band_hz,
                max_peak_shift_hz=max_peak_shift_hz,
                figure_dir=figure_dir,
                write_figure=idx < min_subjects,
            )
        )
    return results
