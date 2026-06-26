"""
src/preprocessing/signal_processor.py
FINAL FIXED VERSION (robust + production-safe)

FIX: _remove_gravity now estimates gravity using a very low-pass Butterworth
     filter (0.1 Hz cut-off) rather than the median of the first fs samples.
     The old approach failed when participants were already moving in the first
     second of a trial, producing a biased gravity estimate and distorted
     gravity-free acceleration channels.

Deterministic: no random sampling or shuffling; output depends only on inputs
and config (see docs/reproducibility.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

from src.preprocessing.unified_bandpass import (
    UnifiedBandpassConfig,
    apply_unified_acc_bandpass,
    lowpass_gyro_columns,
)
from tqdm import tqdm


class SignalProcessor:

    def __init__(self, config: dict):
        self.config = config

        self.proc_dir = Path(config["paths"]["processed_data"])
        self.out_dir  = self.proc_dir / "signals_clean"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pp = config["preprocessing"]

        self.fs      = config["dataset"]["sampling_rate"]
        self.bandpass_cfg = UnifiedBandpassConfig.from_pipeline_config(config)
        self.lp_cut  = float(pp.get("lowpass_cutoff_hz", self.bandpass_cfg.high_hz))
        self.hp_cut  = float(pp.get("highpass_cutoff_hz", self.bandpass_cfg.low_hz))
        self.order   = int(pp.get("lowpass_order", self.bandpass_cfg.order))
        self.beta    = pp["madgwick_beta"]
        self.madgwick_enabled = pp.get("madgwick_enabled", True)
        self.madgwick_sensors = set(pp.get("madgwick_sensors", ["head", "lower_back"]))
        self.gyro_in_degrees = pp.get("gyro_in_degrees", True)
        self.madgwick_use_mag = pp.get("madgwick_use_magnetometer", False)

        self.use_ds_events = pp["gait_event_source"] == "dataset"
        self.hs_threshold_mode = str(
            pp.get("heel_strike_threshold_mode", "prominence")
        ).lower()
        self.hs_peak_percentile = float(pp.get("heel_strike_peak_percentile", 85))
        self.hs_peak_percentile_by_cohort = {
            str(k): float(v)
            for k, v in pp.get("heel_strike_peak_percentile_by_cohort", {}).items()
        }
        floor = pp.get("heel_strike_prominence_floor")
        self.hs_prominence_floor = float(floor) if floor is not None else None
        self.hs_min_interval_s = float(pp.get("heel_strike_min_interval_s", 0.5))
        self.max_nan_fraction = float(pp.get("max_nan_fraction_before_filter", 0.05))
        self.exclude_uturn = bool(pp.get("exclude_uturn_segment", True))
        self.min_walking_segment_s = float(pp.get("min_walking_segment_s", 5.0))
        self._uturn_exclusion_rows: list[dict] = []
        self._trial_cohort: dict[str, str] = {}

        if self.madgwick_enabled:
            try:
                from ahrs.filters import Madgwick as _  # noqa: F401
            except ImportError:
                logger.error(
                    "Madgwick enabled in config but 'ahrs' package is not installed. "
                    "Orientation features will be MISSING. "
                    "Install with: pip install ahrs>=0.3.1"
                )

    # ─────────────────────────────────────────

    def run(self):
        meta_path = self.proc_dir / "trial_metadata.csv"

        if not meta_path.exists():
            raise FileNotFoundError("trial_metadata.csv not found")

        meta = pd.read_csv(meta_path)
        if "trial_id" in meta.columns and "cohort" in meta.columns:
            self._trial_cohort = dict(
                zip(meta["trial_id"].astype(str), meta["cohort"].astype(str))
            )

        has_uturn_cols = {"uturn_start", "uturn_end"}.issubset(meta.columns)

        signals_dir = self.proc_dir / "signals"

        for row in tqdm(
            meta.itertuples(index=False),
            total=len(meta),
            desc="Preprocessing",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            trial_id = row.trial_id

            try:
                cohort = self._trial_cohort.get(str(trial_id))
                uturn_start = int(row.uturn_start) if has_uturn_cols and pd.notna(row.uturn_start) else None
                uturn_end = int(row.uturn_end) if has_uturn_cols and pd.notna(row.uturn_end) else None
                self._process_trial(
                    trial_id,
                    signals_dir,
                    cohort=cohort,
                    uturn_start=uturn_start,
                    uturn_end=uturn_end,
                )
            except Exception as e:
                logger.warning(f"{trial_id} failed: {e}")

        self._write_uturn_exclusion_report()
        logger.info(f"Saved cleaned signals → {self.out_dir}")

    # ─────────────────────────────────────────

    def _process_trial(
        self,
        trial_id: str,
        signals_dir: Path,
        *,
        cohort: str | None = None,
        uturn_start: int | None = None,
        uturn_end: int | None = None,
    ):
        positions = ["head", "lower_back", "left_foot", "right_foot"]
        clean: dict[str, pd.DataFrame] = {}

        for pos in positions:
            path = signals_dir / f"{trial_id}_{pos}.parquet"

            if not path.exists():
                continue

            df = pd.read_parquet(path)

            if df.empty:
                continue

            if (
                self.exclude_uturn
                and uturn_start is not None
                and uturn_end is not None
            ):
                from src.preprocessing.walking_segments import extract_walking_segments

                df, seg_info = extract_walking_segments(
                    df,
                    uturn_start,
                    uturn_end,
                    fs=float(self.fs),
                    min_segment_s=self.min_walking_segment_s,
                )
                if pos == "lower_back":
                    self._uturn_exclusion_rows.append(
                        {"trial_id": trial_id, "sensor": pos, **seg_info}
                    )
                if df is None or df.empty:
                    raise ValueError(
                        f"U-turn exclusion failed ({seg_info.get('status')}): "
                        f"outward={seg_info.get('outward_samples')}, "
                        f"return={seg_info.get('return_samples')}"
                    )

            df = self.process_sensor_dataframe(df, pos, cohort=cohort)
            clean[pos] = df

        if not clean:
            raise ValueError("No valid signals")

        for pos, df in clean.items():
            out = self.out_dir / f"{trial_id}_{pos}.parquet"
            df.to_parquet(out, index=False)

    def _write_uturn_exclusion_report(self) -> None:
        if not self._uturn_exclusion_rows:
            return
        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame(self._uturn_exclusion_rows)
        out_path = metrics_dir / "uturn_exclusion_report.csv"
        report.to_csv(out_path, index=False)
        n_fail = int((report["status"] != "ok").sum()) if "status" in report.columns else 0
        if n_fail:
            logger.warning(
                "U-turn walking extraction failed for {} trials — see {}",
                n_fail,
                out_path,
            )
        else:
            logger.info(
                "U-turn segments excluded from walking signals ({} trials) → {}",
                len(report),
                out_path,
            )

    # ─────────────────────────────────────────

    def process_sensor_dataframe(
        self,
        df: pd.DataFrame,
        sensor_position: str,
        *,
        cohort: str | None = None,
    ) -> pd.DataFrame:
        """
        Single-sensor preprocessing used by batch trials and live API inference.

        Order: filter → resultants → Madgwick orientation (trunk) → gait events (feet)
        → gravity removal (lower back).
        """
        if df.empty:
            return df

        df = self._safe_filter(df)
        if df.empty:
            return df

        df = self._compute_resultant(df)

        if self.madgwick_enabled and sensor_position in self.madgwick_sensors:
            try:
                df = self._attach_orientation(df)
            except ImportError:
                if not getattr(self, "_ahrs_warned", False):
                    logger.error(
                        "ahrs package not installed — orientation features (tilt, roll, pitch) "
                        "will be MISSING for all trials. Install with: pip install ahrs>=0.3.1"
                    )
                    self._ahrs_warned = True

        if not self.use_ds_events and sensor_position == "left_foot":
            df = self._detect_gait_events(df, "left", cohort=cohort)
        elif not self.use_ds_events and sensor_position == "right_foot":
            df = self._detect_gait_events(df, "right", cohort=cohort)

        if sensor_position == "lower_back":
            df = self._remove_gravity(df)

        return df

    @staticmethod
    def _non_finite_fraction(df: pd.DataFrame) -> float:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.0
        values = numeric.to_numpy(dtype=float)
        return float(np.mean(~np.isfinite(values)))

    def _safe_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        nan_frac = self._non_finite_fraction(df)
        if nan_frac > self.max_nan_fraction:
            logger.warning(
                "Discarding sensor segment: {:.1%} non-finite values (>{:.0%} threshold)",
                nan_frac,
                self.max_nan_fraction,
            )
            return pd.DataFrame()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="linear",
                limit_direction="both",
            )

        if len(df) < (self.order * 3):
            return df

        acc_cols = [c for c in df.columns if c.startswith("acc_")]

        try:
            if self.bandpass_cfg.enabled:
                # Stage C: identical 0.5–20 Hz zero-phase bandpass (Voisard + DAPHNET).
                df = apply_unified_acc_bandpass(df, self.bandpass_cfg)
            else:
                gyr_cols = [c for c in df.columns if c.startswith("gyr_")]
                if self.hp_cut > 0:
                    b, a = butter(self.order, self.hp_cut / (self.fs / 2), btype="high")
                    for col in acc_cols:
                        df[col] = filtfilt(b, a, df[col])
                b, a = butter(self.order, self.lp_cut / (self.fs / 2), btype="low")
                for col in acc_cols + gyr_cols:
                    df[col] = filtfilt(b, a, df[col])

            gyr_cols = [c for c in df.columns if c.startswith("gyr_")]
            if gyr_cols and self.bandpass_cfg.enabled:
                df = lowpass_gyro_columns(
                    df,
                    fs_hz=float(self.fs),
                    cutoff_hz=self.bandpass_cfg.high_hz,
                    order=self.bandpass_cfg.order,
                )

        except Exception as e:
            logger.warning(f"Filter failed: {e}")

        return df

    # ─────────────────────────────────────────

    def _compute_resultant(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
            df["acc_resultant"] = np.sqrt(
                df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
            )

        if all(c in df.columns for c in ["gyr_x", "gyr_y", "gyr_z"]):
            df["gyr_resultant"] = np.sqrt(
                df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
            )

        return df

    # ─────────────────────────────────────────

    def _remove_gravity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: estimate the gravity component for each accelerometer axis by
        applying a very low-pass (0.1 Hz) Butterworth filter to the full signal,
        then subtract it to produce gravity-free channels.

        The old approach (median of the first `fs` samples) fails when
        participants are already moving at trial onset, producing a biased
        gravity estimate.  A low-pass filter is more robust because it adapts
        to slow sensor orientation changes throughout the trial.
        """
        df = df.copy()

        # filtfilt needs len(sig) > padlen; 5 s at fs is enough for 0.1 Hz with
        # reduced padding (default padlen assumes very long signals).
        grav_order = 2
        min_len = int(self.fs * 5)
        grav_cutoff = 0.1            # Hz — passes only DC / very slow tilt changes

        nyq = self.fs / 2.0

        for ax in ["acc_x", "acc_y", "acc_z"]:
            if ax not in df.columns:
                continue

            sig = df[ax].values

            if len(sig) >= min_len and grav_cutoff < nyq:
                try:
                    b, a = butter(grav_order, grav_cutoff / nyq, btype="low")
                    padlen = min(3 * (grav_order + 1), len(sig) - 1)
                    gravity = filtfilt(b, a, sig, padlen=padlen)
                except Exception:
                    # Fallback to the global median if filtering fails.
                    gravity = np.full_like(sig, np.median(sig))
            else:
                # Signal too short for 0.1 Hz filter — use robust median.
                gravity = np.full_like(sig, np.median(sig))

            df[f"{ax}_grav_free"] = sig - gravity

        return df

    # ─────────────────────────────────────────

    def detect_heel_strike_indices(
        self,
        df: pd.DataFrame,
        side: str,
        *,
        cohort: str | None = None,
        peak_percentile: float | None = None,
        threshold_mode: str | None = None,
    ) -> np.ndarray:
        """Return sample indices of detected heel strikes (before mutating caller's frame)."""
        marked = self._detect_gait_events(
            df.copy(),
            side,
            cohort=cohort,
            peak_percentile=peak_percentile,
            threshold_mode=threshold_mode,
        )
        col = f"heel_strike_{side}"
        if col not in marked.columns:
            return np.array([], dtype=int)
        return np.where(marked[col].values == 1)[0]

    def _peak_percentile_for_cohort(self, cohort: str | None) -> float:
        if cohort and cohort in self.hs_peak_percentile_by_cohort:
            return self.hs_peak_percentile_by_cohort[cohort]
        return self.hs_peak_percentile

    def _find_heel_strike_indices(
        self,
        acc_v: np.ndarray,
        *,
        cohort: str | None = None,
        peak_percentile: float | None = None,
        threshold_mode: str | None = None,
    ) -> np.ndarray:
        """Detect heel-strike peaks on inverted vertical acceleration.

        ``prominence`` mode (default) ranks peaks by local prominence within the
        trial, reducing sensitivity to global signal amplitude differences across
        pathologies.  ``percentile`` mode keeps the legacy absolute-height rule.
        """
        min_dist = int(self.fs * self.hs_min_interval_s)
        percentile = (
            float(peak_percentile)
            if peak_percentile is not None
            else self._peak_percentile_for_cohort(cohort)
        )
        mode = (threshold_mode or self.hs_threshold_mode).lower()

        if mode == "percentile":
            height = np.percentile(acc_v, percentile)
            hs_idx, _ = find_peaks(acc_v, height=height, distance=min_dist)
            return hs_idx.astype(int)

        if mode != "prominence":
            raise ValueError(
                f"Unknown heel_strike_threshold_mode '{mode}' "
                "(expected 'prominence' or 'percentile')"
            )

        candidates, props = find_peaks(acc_v, distance=min_dist, prominence=0)
        if len(candidates) == 0:
            return np.array([], dtype=int)

        prom = props["prominences"]
        min_prom = float(np.percentile(prom, percentile))
        if self.hs_prominence_floor is not None:
            min_prom = max(min_prom, self.hs_prominence_floor)
        return candidates[prom >= min_prom].astype(int)

    def _detect_gait_events(
        self,
        df: pd.DataFrame,
        side: str,
        *,
        cohort: str | None = None,
        peak_percentile: float | None = None,
        threshold_mode: str | None = None,
    ) -> pd.DataFrame:
        """
        Heel-strike peaks on inverted vertical acceleration.

        Threshold mode and percentile are configurable; cohort-specific
        percentiles can be set in ``heel_strike_peak_percentile_by_cohort``.
        Tune per-cohort values with ``validate_gait_events`` when ground truth
        is available (see ``gait_event_validation.tune_cohort_percentiles``).
        """
        if "acc_z" not in df.columns:
            return df

        acc_v = -df["acc_z"].values
        hs_idx = self._find_heel_strike_indices(
            acc_v,
            cohort=cohort,
            peak_percentile=peak_percentile,
            threshold_mode=threshold_mode,
        )

        min_dist = int(self.fs * self.hs_min_interval_s)
        to_signal = df["acc_resultant"].values if "acc_resultant" in df.columns else acc_v
        to_idx, _ = find_peaks(-to_signal, distance=min_dist)

        df[f"heel_strike_{side}"] = 0
        df[f"toe_off_{side}"]     = 0

        if len(hs_idx):
            df.loc[df.index[hs_idx], f"heel_strike_{side}"] = 1
        if len(to_idx):
            df.loc[df.index[to_idx], f"toe_off_{side}"] = 1

        return df

    # ─────────────────────────────────────────
    # Madgwick sensor fusion (ahrs)
    # ─────────────────────────────────────────

    def compute_orientation(self, df: pd.DataFrame) -> np.ndarray:
        """Quaternion time series [w, x, y, z] via Madgwick IMU fusion."""
        req = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        n = len(df)
        if not all(c in df.columns for c in req) or n == 0:
            return np.tile([1.0, 0.0, 0.0, 0.0], (max(n, 1), 1))

        from ahrs.filters import Madgwick

        acc = df[["acc_x", "acc_y", "acc_z"]].values.astype(float)
        gyr = df[["gyr_x", "gyr_y", "gyr_z"]].values.astype(float)
        if self.gyro_in_degrees:
            gyr = np.deg2rad(gyr)

        madgwick = Madgwick(frequency=float(self.fs), beta=float(self.beta))
        quats = np.zeros((n, 4), dtype=float)
        quats[0] = np.array([1.0, 0.0, 0.0, 0.0])

        mag = None
        if self.madgwick_use_mag and all(
            c in df.columns for c in ["mag_x", "mag_y", "mag_z"]
        ):
            mag = df[["mag_x", "mag_y", "mag_z"]].values.astype(float)

        for i in range(1, n):
            if mag is not None:
                quats[i] = madgwick.updateMARG(quats[i - 1], gyr[i], acc[i], mag[i])
            else:
                quats[i] = madgwick.updateIMU(quats[i - 1], gyr[i], acc[i])

        return quats

    @staticmethod
    def _euler_tilt_from_quaternions(quats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scalar-first quaternions → roll, pitch, tilt-from-vertical (radians)."""
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)
        tilt = np.arctan2(np.sqrt(gx * gx + gy * gy), np.clip(gz, 1e-9, None))

        return roll, pitch, tilt

    def _attach_orientation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append quaternion and trunk-orientation columns for feature extraction."""
        df = df.copy()
        quats = self.compute_orientation(df)
        roll, pitch, tilt = self._euler_tilt_from_quaternions(quats)

        df["quat_w"] = quats[:, 0]
        df["quat_x"] = quats[:, 1]
        df["quat_y"] = quats[:, 2]
        df["quat_z"] = quats[:, 3]
        df["roll_rad"] = roll
        df["pitch_rad"] = pitch
        df["tilt_rad"] = tilt
        df["tilt_deg"] = np.rad2deg(tilt)
        return df