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


class SignalProcessor:

    def __init__(self, config: dict):
        self.config = config

        self.proc_dir = Path(config["paths"]["processed_data"])
        self.out_dir  = self.proc_dir / "signals_clean"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pp = config["preprocessing"]

        self.fs      = config["dataset"]["sampling_rate"]
        self.lp_cut  = pp["lowpass_cutoff_hz"]
        self.hp_cut  = pp["highpass_cutoff_hz"]
        self.order   = pp["lowpass_order"]
        self.beta    = pp["madgwick_beta"]
        self.madgwick_enabled = pp.get("madgwick_enabled", True)
        self.madgwick_sensors = set(pp.get("madgwick_sensors", ["head", "lower_back"]))
        self.gyro_in_degrees = pp.get("gyro_in_degrees", True)
        self.madgwick_use_mag = pp.get("madgwick_use_magnetometer", False)

        self.use_ds_events = pp["gait_event_source"] == "dataset"

    # ─────────────────────────────────────────

    def run(self):
        meta_path = self.proc_dir / "trial_metadata.csv"

        if not meta_path.exists():
            raise FileNotFoundError("trial_metadata.csv not found")

        meta = pd.read_csv(meta_path)
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
                self._process_trial(trial_id, signals_dir)
            except Exception as e:
                logger.warning(f"{trial_id} failed: {e}")

        logger.info(f"Saved cleaned signals → {self.out_dir}")

    # ─────────────────────────────────────────

    def _process_trial(self, trial_id: str, signals_dir: Path):
        positions = ["head", "lower_back", "left_foot", "right_foot"]
        clean: dict[str, pd.DataFrame] = {}

        for pos in positions:
            path = signals_dir / f"{trial_id}_{pos}.parquet"

            if not path.exists():
                continue

            df = pd.read_parquet(path)

            if df.empty:
                continue

            df = self.process_sensor_dataframe(df, pos)
            clean[pos] = df

        if not clean:
            raise ValueError("No valid signals")

        for pos, df in clean.items():
            out = self.out_dir / f"{trial_id}_{pos}.parquet"
            df.to_parquet(out, index=False)

    # ─────────────────────────────────────────

    def process_sensor_dataframe(self, df: pd.DataFrame, sensor_position: str) -> pd.DataFrame:
        """
        Single-sensor preprocessing used by batch trials and live API inference.

        Order: filter → resultants → Madgwick orientation (trunk) → gait events (feet)
        → gravity removal (lower back).
        """
        if df.empty:
            return df

        df = self._safe_filter(df)
        df = self._compute_resultant(df)

        if self.madgwick_enabled and sensor_position in self.madgwick_sensors:
            df = self._attach_orientation(df)

        if not self.use_ds_events and sensor_position == "left_foot":
            df = self._detect_gait_events(df, "left")
        elif not self.use_ds_events and sensor_position == "right_foot":
            df = self._detect_gait_events(df, "right")

        if sensor_position == "lower_back":
            df = self._remove_gravity(df)

        return df

    def _safe_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.interpolate().bfill().ffill()

        if len(df) < (self.order * 3):
            return df

        acc_cols = [c for c in df.columns if c.startswith("acc_")]
        gyr_cols = [c for c in df.columns if c.startswith("gyr_")]

        try:
            if self.hp_cut > 0:
                b, a = butter(self.order, self.hp_cut / (self.fs / 2), btype="high")
                for col in acc_cols:
                    df[col] = filtfilt(b, a, df[col])

            b, a = butter(self.order, self.lp_cut / (self.fs / 2), btype="low")

            for col in acc_cols + gyr_cols:
                df[col] = filtfilt(b, a, df[col])

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

        # Minimum signal length required for filtfilt with this order/cutoff.
        min_len = int(self.fs * 10)  # ~10 seconds needed for stable 0.1 Hz filter
        grav_cutoff = 0.1            # Hz — passes only DC / very slow tilt changes

        nyq = self.fs / 2.0

        for ax in ["acc_x", "acc_y", "acc_z"]:
            if ax not in df.columns:
                continue

            sig = df[ax].values

            if len(sig) >= min_len and grav_cutoff < nyq:
                try:
                    b, a = butter(2, grav_cutoff / nyq, btype="low")
                    gravity = filtfilt(b, a, sig)
                except Exception:
                    # Fallback to the global median if filtering fails.
                    gravity = np.full_like(sig, np.median(sig))
            else:
                # Signal too short for 0.1 Hz filter — use robust median.
                gravity = np.full_like(sig, np.median(sig))

            df[f"{ax}_grav_free"] = sig - gravity

        return df

    # ─────────────────────────────────────────

    def detect_heel_strike_indices(self, df: pd.DataFrame, side: str) -> np.ndarray:
        """Return sample indices of detected heel strikes (before mutating caller's frame)."""
        marked = self._detect_gait_events(df.copy(), side)
        col = f"heel_strike_{side}"
        if col not in marked.columns:
            return np.array([], dtype=int)
        return np.where(marked[col].values == 1)[0]

    def _detect_gait_events(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Heel-strike peaks on inverted vertical acceleration (75th-percentile height).

        Validated against Figshare annotations via ``validate_gait_events`` stage.
        """
        if "acc_z" not in df.columns:
            return df

        acc_v = -df["acc_z"].values

        height = np.percentile(acc_v, 75)

        hs_idx, _ = find_peaks(acc_v, height=height, distance=int(self.fs * 0.3))

        to_signal = df["acc_resultant"].values if "acc_resultant" in df.columns else acc_v
        to_idx, _ = find_peaks(-to_signal, distance=int(self.fs * 0.3))

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