"""
src/preprocessing/signal_processor.py
FINAL FIXED VERSION (robust + production-safe)

FIX: _remove_gravity now estimates gravity using a very low-pass Butterworth
     filter (0.1 Hz cut-off) rather than the median of the first fs samples.
     The old approach failed when participants were already moving in the first
     second of a trial, producing a biased gravity estimate and distorted
     gravity-free acceleration channels.
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

        self.use_ds_events = pp["gait_event_source"] == "dataset"

    # ─────────────────────────────────────────

    def run(self):
        meta_path = self.proc_dir / "trial_metadata.csv"

        if not meta_path.exists():
            raise FileNotFoundError("trial_metadata.csv not found")

        meta = pd.read_csv(meta_path)
        signals_dir = self.proc_dir / "signals"

        for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Preprocessing"):
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

            df = self._safe_filter(df)
            df = self._compute_resultant(df)

            clean[pos] = df

        if not clean:
            raise ValueError("No valid signals")

        if not self.use_ds_events:
            if "left_foot" in clean:
                clean["left_foot"]  = self._detect_gait_events(clean["left_foot"],  "left")
            if "right_foot" in clean:
                clean["right_foot"] = self._detect_gait_events(clean["right_foot"], "right")

        if "lower_back" in clean:
            clean["lower_back"] = self._remove_gravity(clean["lower_back"])

        for pos, df in clean.items():
            out = self.out_dir / f"{trial_id}_{pos}.parquet"
            df.to_parquet(out, index=False)

    # ─────────────────────────────────────────

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

    def _detect_gait_events(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
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
    # OPTIONAL: Madgwick (kept for future use)
    # ─────────────────────────────────────────

    def compute_orientation(self, df: pd.DataFrame) -> np.ndarray:
        req = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

        if not all(c in df.columns for c in req):
            return np.tile([1, 0, 0, 0], (len(df), 1))

        dt  = 1.0 / self.fs
        q   = np.array([1.0, 0.0, 0.0, 0.0])
        quats = []

        acc = df[["acc_x", "acc_y", "acc_z"]].values
        gyr = df[["gyr_x", "gyr_y", "gyr_z"]].values

        for i in range(len(df)):
            q = self._madgwick_step(q, acc[i], gyr[i], dt)
            quats.append(q.copy())

        return np.array(quats)