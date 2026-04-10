"""
src/features/feature_extractor.py
===================================
Stage 4: Extract temporal, spatial, spectral, trunk-dynamics, and asymmetry
features from preprocessed IMU signals.

Output:
  data/features/trial_features.parquet    — one row per trial
  data/features/patient_features.parquet  — aggregated to patient level
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import welch
from scipy.stats import entropy as sp_entropy
from tqdm import tqdm


class FeatureExtractor:

    def __init__(self, config: dict):
        self.config   = config
        self.proc_dir = Path(config["paths"]["processed_data"])
        self.out_dir  = Path(config["paths"]["features"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fs = config["dataset"]["sampling_rate"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self):
        meta = pd.read_csv(self.proc_dir / "trial_metadata.csv")
        rows = []

        for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Extracting features"):
            feats = self._extract_trial(row._asdict())
            if feats:
                rows.append(feats)

        if not rows:
            logger.warning("No features extracted. Ensure preprocessing was run first.")
            return

        trial_df = pd.DataFrame(rows)

        # Save trial-level features
        trial_path = self.out_dir / "trial_features.parquet"
        trial_df.to_parquet(trial_path, index=False)
        logger.info(f"Trial features saved → {trial_path}  shape={trial_df.shape}")

        # Aggregate to patient level
        patient_df = self._aggregate_to_patient(trial_df)
        patient_path = self.out_dir / "patient_features.parquet"
        patient_df.to_parquet(patient_path, index=False)
        logger.info(f"Patient features saved → {patient_path}  shape={patient_df.shape}")

    # ── Per-trial extraction ───────────────────────────────────────────────────

    def _extract_trial(self, row: pd.Series) -> Optional[dict]:
        trial_id = row["trial_id"]
        signals  = self._load_signals(trial_id)
        if not signals:
            return None

        feats: dict = {
            "trial_id":         trial_id,
            "participant_id":   row["participant_id"],
            "cohort":           row["cohort"],
            "fall_probability": row["fall_probability"],
            "risk_label":       row["risk_label"],
            "laterality_biased": row.get("laterality_biased", False),
        }

        # Lower-back features (primary)
        lb = signals.get("lower_back")
        if lb is not None:
            feats.update(self._trunk_dynamics(lb))
            feats.update(self._spectral_features(lb, prefix="lb"))

        # Foot-derived gait cycle features
        lf = signals.get("left_foot")
        rf = signals.get("right_foot")
        if lf is not None and rf is not None:
            feats.update(self._gait_cycle_features(lf, rf))
            feats.update(self._asymmetry_features(lf, rf))

        # Head features (postural control)
        hd = signals.get("head")
        if hd is not None:
            feats.update(self._trunk_dynamics(hd, prefix="head"))

        return feats

    # ── Feature groups ─────────────────────────────────────────────────────────

    def _trunk_dynamics(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        f = {}
        for ax, label in [("acc_x", "ap"), ("acc_y", "ml"), ("acc_z", "v")]:
            if ax in df.columns:
                sig = df[ax].values
                f[f"{prefix}_rms_{label}"]      = float(np.sqrt(np.mean(sig ** 2)))
                f[f"{prefix}_range_{label}"]    = float(np.ptp(sig))
                f[f"{prefix}_std_{label}"]      = float(np.std(sig))
                # Jerk (derivative of acceleration)
                jerk = np.diff(sig) * self.fs
                f[f"{prefix}_jerk_mean_{label}"] = float(np.mean(np.abs(jerk)))
                f[f"{prefix}_jerk_max_{label}"]  = float(np.max(np.abs(jerk)))

        # Total resultant
        if "acc_resultant" in df.columns:
            res = df["acc_resultant"].values
            f[f"{prefix}_rms_total"]  = float(np.sqrt(np.mean(res ** 2)))
            f[f"{prefix}_lyapunov"]   = self._lyapunov_exponent(res)

        return f

    def _spectral_features(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        f = {}
        if "acc_z" not in df.columns:
            return f
        sig = df["acc_z"].values
        freqs, pxx = welch(sig, fs=self.fs, nperseg=min(256, len(sig) // 2))

        if len(pxx) == 0:
            return f

        dom_idx = np.argmax(pxx)
        f[f"{prefix}_dominant_freq"]    = float(freqs[dom_idx])
        f[f"{prefix}_spectral_entropy"] = float(sp_entropy(pxx / (pxx.sum() + 1e-10)))

        # Band power
        def band_power(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not mask.any():
                return 0.0
            # Simple numerical integration using trapezoidal rule
            freq_masked = freqs[mask]
            pxx_masked = pxx[mask]
            if len(freq_masked) > 1:
                return float(np.sum(0.5 * (pxx_masked[:-1] + pxx_masked[1:]) * np.diff(freq_masked)))
            else:
                return 0.0

        f[f"{prefix}_power_0_1hz"]  = band_power(0, 1)
        f[f"{prefix}_power_1_3hz"]  = band_power(1, 3)
        f[f"{prefix}_power_3_10hz"] = band_power(3, 10)

        # Harmonic ratio (ratio of even/odd harmonics of step frequency)
        step_freq = f.get(f"{prefix}_dominant_freq", 1.0)
        if step_freq > 0:
            harmonics = [pxx[np.argmin(np.abs(freqs - step_freq * k))] for k in range(1, 7)]
            even_sum = sum(harmonics[1::2])  # 2nd, 4th, 6th
            odd_sum  = sum(harmonics[0::2])  # 1st, 3rd, 5th
            f[f"{prefix}_harmonic_ratio"] = float(even_sum / (odd_sum + 1e-10))

        return f

    def _gait_cycle_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        f = {}
        # Use heel-strike columns if available
        for df_foot, side in [(lf, "left"), (rf, "right")]:
            hs_col = f"heel_strike_{side}"
            to_col = f"toe_off_{side}"
            if hs_col in df_foot.columns:
                hs_idx = np.where(df_foot[hs_col].values == 1)[0]
                if len(hs_idx) >= 3:
                    stride_times = np.diff(hs_idx) / self.fs
                    f[f"{side}_stride_time_mean"] = float(np.mean(stride_times))
                    f[f"{side}_stride_time_std"]  = float(np.std(stride_times))
                    f[f"{side}_stride_time_cv"]   = float(
                        np.std(stride_times) / (np.mean(stride_times) + 1e-10)
                    )
                    f[f"{side}_cadence"] = float(60.0 / (np.mean(stride_times) + 1e-10))
                    f[f"{side}_step_count"] = len(hs_idx)

            if to_col in df_foot.columns and hs_col in df_foot.columns:
                hs_idx = np.where(df_foot[hs_col].values == 1)[0]
                to_idx = np.where(df_foot[to_col].values == 1)[0]
                if len(hs_idx) > 1 and len(to_idx) > 0:
                    # Estimate stance / swing phase ratios
                    stances, swings = [], []
                    for hs in hs_idx[:-1]:
                        toe_offs_after = to_idx[to_idx > hs]
                        if len(toe_offs_after) > 0:
                            stance = (toe_offs_after[0] - hs) / self.fs
                            stances.append(stance)
                    if stances:
                        f[f"{side}_stance_ratio"] = float(
                            np.mean(stances) / (f.get(f"{side}_stride_time_mean", 1.0) + 1e-10)
                        )

        # Cadence (use mean of both feet if available)
        cads = [f.get("left_cadence"), f.get("right_cadence")]
        cads = [c for c in cads if c is not None]
        if cads:
            f["cadence_mean"] = float(np.mean(cads))

        return f

    def _asymmetry_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        f = {}
        # Stride-time asymmetry based on heel-strike derived stride time (if available)
        if "heel_strike_left" in lf.columns:
            hs_l = np.where(lf["heel_strike_left"].values == 1)[0]
            if len(hs_l) >= 3:
                stride_times_l = np.diff(hs_l) / self.fs
                st_l_mean = float(np.mean(stride_times_l))
                st_l_std = float(np.std(stride_times_l))
            else:
                st_l_mean = st_l_std = None
        else:
            st_l_mean = st_l_std = None

        if "heel_strike_right" in rf.columns:
            hs_r = np.where(rf["heel_strike_right"].values == 1)[0]
            if len(hs_r) >= 3:
                stride_times_r = np.diff(hs_r) / self.fs
                st_r_mean = float(np.mean(stride_times_r))
                st_r_std = float(np.std(stride_times_r))
            else:
                st_r_mean = st_r_std = None
        else:
            st_r_mean = st_r_std = None

        if st_l_mean is not None and st_r_mean is not None:
            f["stride_time_mean_asymmetry"] = float(
                abs(st_l_mean - st_r_mean) / (st_l_mean + st_r_mean + 1e-10)
            )

        if st_l_std is not None and st_r_std is not None:
            f["stride_time_std_asymmetry"] = float(
                abs(st_l_std - st_r_std) / (st_l_std + st_r_std + 1e-10)
            )

        # Compute on resultant acc
        if "acc_resultant" in lf.columns and "acc_resultant" in rf.columns:
            n = min(len(lf), len(rf))
            l_rms = float(np.sqrt(np.mean(lf["acc_resultant"].values[:n] ** 2)))
            r_rms = float(np.sqrt(np.mean(rf["acc_resultant"].values[:n] ** 2)))
            f["asymmetry_rms_acc"] = float(abs(l_rms - r_rms) / (l_rms + r_rms + 1e-10))

        # Heel-strike timing asymmetry
        hs_l = np.where(lf["heel_strike_left"].values == 1)[0] if "heel_strike_left" in lf.columns else np.array([])
        hs_r = np.where(rf["heel_strike_right"].values == 1)[0] if "heel_strike_right" in rf.columns else np.array([])
        if len(hs_l) >= 2 and len(hs_r) >= 2:
            st_l = np.mean(np.diff(hs_l)) / self.fs
            st_r = np.mean(np.diff(hs_r)) / self.fs
            f["stride_time_asymmetry"] = float(abs(st_l - st_r) / (st_l + st_r + 1e-10))

        return f

    def _lyapunov_exponent(self, signal: np.ndarray, m: int = 5, tau: int = 10) -> float:
        """
        Approximate largest Lyapunov exponent (local dynamic stability).
        Simplified Rosenstein algorithm.
        """
        N = len(signal)
        if N < m * tau + 50:
            return float("nan")
        # Embed
        n_pts = N - (m - 1) * tau
        emb = np.array([[signal[i + j * tau] for j in range(m)] for i in range(n_pts)])
        divs = []
        for i in range(min(n_pts, 200)):
            dists = np.linalg.norm(emb - emb[i], axis=1)
            dists[max(0, i - 10):i + 10] = np.inf
            j = np.argmin(dists)
            steps = min(20, n_pts - max(i, j) - 1)
            if steps > 0:
                future_dists = [
                    np.linalg.norm(emb[i + s] - emb[j + s]) + 1e-10
                    for s in range(steps)
                ]
                divs.append(np.mean(np.log(future_dists)))

        return float(np.mean(divs)) if divs else float("nan")

    # ── Patient aggregation ────────────────────────────────────────────────────

    def _aggregate_to_patient(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trial-level features to patient level.
        For each numeric feature: compute mean and std across trials.
        Strategy: 'mean_std' as per config.
        """
        meta_cols  = ["participant_id", "cohort", "risk_label", "fall_probability", "laterality_biased"]
        feat_cols  = [c for c in trial_df.columns if c not in meta_cols + ["trial_id"]]
        num_cols   = trial_df[feat_cols].select_dtypes(include=np.number).columns.tolist()

        patient_rows = []
        for pid, grp in trial_df.groupby("participant_id"):
            row: dict = {
                "participant_id":    pid,
                "cohort":            grp["cohort"].iloc[0],
                "risk_label":        grp["risk_label"].iloc[0],
                "fall_probability":  grp["fall_probability"].iloc[0],  # NEW: include probability
                "laterality_biased": grp["laterality_biased"].iloc[0],
                "n_trials":          len(grp),
            }
            for col in num_cols:
                vals = grp[col].dropna().values
                if len(vals) > 0:
                    row[f"{col}_mean"] = float(np.mean(vals))
                    row[f"{col}_std"]  = float(np.std(vals))
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"]  = np.nan

            patient_rows.append(row)

        return pd.DataFrame(patient_rows)

    # ── IO helper ──────────────────────────────────────────────────────────────

    def _load_signals(self, trial_id: str) -> dict[str, pd.DataFrame]:
        signals = {}
        for pos in ["head", "lower_back", "left_foot", "right_foot"]:
            for subdir in ["signals_clean", "signals"]:
                p = self.proc_dir / subdir / f"{trial_id}_{pos}.parquet"
                if p.exists():
                    signals[pos] = pd.read_parquet(p)
                    break
        return signals
