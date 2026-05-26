"""
src/features/feature_extractor.py
===================================
Stage 4: Extract temporal gait-cycle, spectral, trunk-dynamics, orientation,
nonlinear dynamics, and asymmetry features from preprocessed IMU signals.

Absolute spatial metrics (step length, gait speed, step width) are not extracted;
see docs/spatial_features.md.
Patient rows aggregate each trial feature with mean, std, range, and
session-ordered linear trend across trials.

Output:
  data/features/trial_features.parquet    — one row per trial
  data/features/patient_features.parquet  — aggregated to patient level

LEAKAGE POLICY
--------------
The columns listed in METADATA_ONLY_COLS (cohort, risk_label, fall_probability,
laterality_biased) are label-derived and must never appear in the feature
matrix used for model training. They are carried through the pipeline as
bookkeeping columns only and are explicitly excluded from numeric feature
aggregation in _aggregate_to_patient.
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

# Import the authoritative set of label-derived columns from data_loader so
# there is a single source of truth across the whole pipeline.
from src.dataset.label_policy import multiclass_label_from_cohort
from src.ingestion.data_loader import METADATA_ONLY_COLS
from src.features.nonlinear_metrics import (
    approximate_entropy,
    largest_lyapunov_exponent,
)
from src.features.patient_temporal_aggregation import (
    aggregate_trial_values,
    order_trial_group,
    parse_patient_aggregation_config,
)

# All columns that are metadata / bookkeeping and must never be model features.
_META_COLS = METADATA_ONLY_COLS | {"trial_id", "participant_id", "n_trials", "session"}


def spectral_centroid_hz(freqs: np.ndarray, pxx: np.ndarray) -> float:
    """First moment of the PSD: sum(f * P) / sum(P) in Hz."""
    pxx = np.asarray(pxx, dtype=float)
    total = float(pxx.sum())
    if total < 1e-12:
        return float("nan")
    return float(np.sum(np.asarray(freqs, dtype=float) * pxx) / total)


class FeatureExtractor:

    def __init__(self, config: dict):
        self.config   = config
        self.proc_dir = Path(config["paths"]["processed_data"])
        self.out_dir  = Path(config["paths"]["features"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fs = config["dataset"]["sampling_rate"]
        feat_cfg = config.get("features", {})
        self._lyap_cfg = feat_cfg.get("lyapunov", {})
        self._apen_cfg = feat_cfg.get("approximate_entropy", {})
        self._patient_agg_cfg = parse_patient_aggregation_config(feat_cfg)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self):
        meta_path = self.proc_dir / "trial_metadata.csv"
        if not meta_path.exists():
            logger.error(f"trial_metadata.csv not found at {meta_path}. Run preprocessing first.")
            return

        meta = pd.read_csv(meta_path)
        rows = []

        for row in tqdm(
            meta.itertuples(index=False),
            total=len(meta),
            desc="Extracting features",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            feats = self._extract_trial(row._asdict())
            if feats:
                rows.append(feats)

        if not rows:
            logger.warning("No features extracted. Ensure preprocessing was run first.")
            return

        trial_df = pd.DataFrame(rows)

        trial_path = self.out_dir / "trial_features.parquet"
        trial_df.to_parquet(trial_path, index=False)
        logger.info(f"Trial features saved → {trial_path}  shape={trial_df.shape}")

        patient_df = self._aggregate_to_patient(trial_df)
        patient_path = self.out_dir / "patient_features.parquet"
        patient_df.to_parquet(patient_path, index=False)
        logger.info(f"Patient features saved → {patient_path}  shape={patient_df.shape}")

    # ── Per-trial extraction ───────────────────────────────────────────────────

    def _extract_trial(self, row: dict) -> Optional[dict]:
        trial_id = row["trial_id"]
        signals  = self._load_signals(trial_id)
        if not signals:
            return None

        # FIX: only bookkeeping columns here — none of these must reach the
        # model feature matrix. _aggregate_to_patient explicitly excludes
        # everything in _META_COLS from numeric aggregation.
        feats: dict = {
            "trial_id":       trial_id,
            "session":        row.get("session"),
            "participant_id": row["participant_id"],
            # Metadata-only — label-derived, excluded from feature aggregation:
            "cohort":            row["cohort"],
            "risk_label":        row["risk_label"],
            "multiclass_label":  row.get(
                "multiclass_label",
                multiclass_label_from_cohort(str(row["cohort"])),
            ),
            "fall_probability":  row.get("fall_probability"),
            "laterality_biased": row.get("laterality_biased", False),
        }

        lb = signals.get("lower_back")
        if lb is not None:
            feats.update(self._trunk_dynamics(lb, prefix="lb"))
            feats.update(self._spectral_features(lb, prefix="lb"))
            feats.update(self._orientation_features(lb, prefix="lb"))

        lf = signals.get("left_foot")
        rf = signals.get("right_foot")
        if lf is not None and rf is not None:
            feats.update(self._gait_cycle_features(lf, rf))
            feats.update(self._asymmetry_features(lf, rf))

        hd = signals.get("head")
        if hd is not None:
            feats.update(self._trunk_dynamics(hd, prefix="head"))
            feats.update(self._orientation_features(hd, prefix="head"))

        return feats

    # ── Feature groups ─────────────────────────────────────────────────────────

    def _trunk_dynamics(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        """Compute time-domain trunk-dynamics features.

        FIX: prefers gravity-free axes (acc_x_grav_free etc.) produced by the
        signal processor when available, falling back to raw axes. Using raw
        axes includes the gravity DC component in RMS and range, inflating
        those features and making them dependent on sensor orientation.
        """
        f: dict = {}

        axis_map = [
            ("acc_x_grav_free", "acc_x", "ap"),
            ("acc_y_grav_free", "acc_y", "ml"),
            ("acc_z_grav_free", "acc_z", "v"),
        ]

        for grav_free_col, raw_col, label in axis_map:
            col = grav_free_col if grav_free_col in df.columns else raw_col
            if col not in df.columns:
                continue

            sig = df[col].values
            f[f"{prefix}_rms_{label}"]       = float(np.sqrt(np.mean(sig ** 2)))
            f[f"{prefix}_range_{label}"]     = float(np.ptp(sig))
            f[f"{prefix}_std_{label}"]       = float(np.std(sig))
            jerk = np.diff(sig) * self.fs
            f[f"{prefix}_jerk_mean_{label}"] = float(np.mean(np.abs(jerk)))
            f[f"{prefix}_jerk_max_{label}"]  = float(np.max(np.abs(jerk)))

        if "acc_resultant" in df.columns:
            res = df["acc_resultant"].values
            f[f"{prefix}_rms_total"] = float(np.sqrt(np.mean(res ** 2)))
            f[f"{prefix}_lyapunov"] = largest_lyapunov_exponent(res, self._lyap_cfg)
            f[f"{prefix}_apen"] = approximate_entropy(res, self._apen_cfg)

        return f

    def _orientation_features(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        """Trunk tilt and postural stability from Madgwick quaternion fusion."""
        if "tilt_rad" not in df.columns:
            return {}

        tilt_deg = np.rad2deg(df["tilt_rad"].values.astype(float))
        pitch_deg = np.rad2deg(df["pitch_rad"].values.astype(float))
        roll_deg = np.rad2deg(df["roll_rad"].values.astype(float))

        tilt_rate = np.diff(tilt_deg) * self.fs
        sway_vel = float(np.mean(np.abs(tilt_rate))) if len(tilt_rate) else float("nan")

        return {
            f"{prefix}_tilt_mean_deg": float(np.mean(tilt_deg)),
            f"{prefix}_tilt_std_deg": float(np.std(tilt_deg)),
            f"{prefix}_tilt_range_deg": float(np.ptp(tilt_deg)),
            f"{prefix}_pitch_std_deg": float(np.std(pitch_deg)),
            f"{prefix}_roll_std_deg": float(np.std(roll_deg)),
            f"{prefix}_postural_sway_vel": sway_vel,
        }

    def _spectral_features(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        """Compute frequency-domain features.

        Welch PSD on lower-back vertical acceleration (prefers acc_z_grav_free).
        Outputs: dominant frequency, spectral centroid, entropy, band power,
        and harmonic ratio.
        """
        f: dict = {}

        # Prefer gravity-free vertical axis.
        sig_col = (
            "acc_z_grav_free" if "acc_z_grav_free" in df.columns
            else "acc_z" if "acc_z" in df.columns
            else None
        )
        if sig_col is None:
            return f

        sig = df[sig_col].values
        nperseg = min(256, max(4, len(sig) // 2))
        freqs, pxx = welch(sig, fs=self.fs, nperseg=nperseg)

        if len(pxx) == 0 or pxx.sum() < 1e-12:
            return f

        dom_idx = int(np.argmax(pxx))
        dominant_freq = float(freqs[dom_idx])

        f[f"{prefix}_dominant_freq"]      = dominant_freq
        f[f"{prefix}_spectral_centroid"]  = spectral_centroid_hz(freqs, pxx)
        f[f"{prefix}_spectral_entropy"]   = float(sp_entropy(pxx / (pxx.sum() + 1e-10)))

        def band_power(f_lo: float, f_hi: float) -> float:
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

        f[f"{prefix}_power_0_1hz"]  = band_power(0.0, 1.0)
        f[f"{prefix}_power_1_3hz"]  = band_power(1.0, 3.0)
        f[f"{prefix}_power_3_10hz"] = band_power(3.0, 10.0)

        # Harmonic ratio.
        # FIX: use the already-computed dominant_freq directly instead of
        # re-fetching from the dict with a misleading .get() + default.
        if dominant_freq > 0:
            harmonics = [
                float(pxx[np.argmin(np.abs(freqs - dominant_freq * k))])
                for k in range(1, 7)
            ]
            even_sum = sum(harmonics[1::2])  # 2nd, 4th, 6th harmonics
            odd_sum  = sum(harmonics[0::2])  # 1st, 3rd, 5th harmonics
            f[f"{prefix}_harmonic_ratio"] = float(even_sum / (odd_sum + 1e-10))

        return f

    def _gait_cycle_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        f: dict = {}

        for df_foot, side in [(lf, "left"), (rf, "right")]:
            hs_col = f"heel_strike_{side}"
            to_col = f"toe_off_{side}"

            if hs_col in df_foot.columns:
                hs_idx = np.where(df_foot[hs_col].values == 1)[0]
                if len(hs_idx) >= 3:
                    stride_times = np.diff(hs_idx) / self.fs
                    st_mean = float(np.mean(stride_times))
                    f[f"{side}_stride_time_mean"] = st_mean
                    f[f"{side}_stride_time_std"]  = float(np.std(stride_times))
                    f[f"{side}_stride_time_cv"]   = float(
                        np.std(stride_times) / (st_mean + 1e-10)
                    )
                    f[f"{side}_cadence"]    = float(60.0 / (st_mean + 1e-10))
                    f[f"{side}_step_count"] = int(len(hs_idx))

                if to_col in df_foot.columns and len(hs_idx) > 1:
                    to_idx = np.where(df_foot[to_col].values == 1)[0]
                    stances = []
                    for hs in hs_idx[:-1]:
                        toe_offs_after = to_idx[to_idx > hs]
                        if len(toe_offs_after) > 0:
                            stances.append((toe_offs_after[0] - hs) / self.fs)
                    if stances:
                        st_mean_ref = f.get(f"{side}_stride_time_mean", 1.0)
                        f[f"{side}_stance_ratio"] = float(
                            np.mean(stances) / (st_mean_ref + 1e-10)
                        )

        cads = [f[k] for k in ("left_cadence", "right_cadence") if k in f]
        if cads:
            f["cadence_mean"] = float(np.mean(cads))

        return f

    def _asymmetry_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        """Compute bilateral asymmetry features.

        FIX: removed the duplicate stride_time_asymmetry computation that
        shadowed stride_time_mean_asymmetry with an identically-derived value,
        inflating feature count with redundant information.
        """
        f: dict = {}

        def _stride_stats(df_foot: pd.DataFrame, side: str):
            col = f"heel_strike_{side}"
            if col not in df_foot.columns:
                return None, None
            hs_idx = np.where(df_foot[col].values == 1)[0]
            if len(hs_idx) < 3:
                return None, None
            st = np.diff(hs_idx) / self.fs
            return float(np.mean(st)), float(np.std(st))

        st_l_mean, st_l_std = _stride_stats(lf, "left")
        st_r_mean, st_r_std = _stride_stats(rf, "right")

        if st_l_mean is not None and st_r_mean is not None:
            f["stride_time_mean_asymmetry"] = float(
                abs(st_l_mean - st_r_mean) / (st_l_mean + st_r_mean + 1e-10)
            )

        if st_l_std is not None and st_r_std is not None:
            f["stride_time_std_asymmetry"] = float(
                abs(st_l_std - st_r_std) / (st_l_std + st_r_std + 1e-10)
            )

        if "acc_resultant" in lf.columns and "acc_resultant" in rf.columns:
            n = min(len(lf), len(rf))
            l_rms = float(np.sqrt(np.mean(lf["acc_resultant"].values[:n] ** 2)))
            r_rms = float(np.sqrt(np.mean(rf["acc_resultant"].values[:n] ** 2)))
            f["asymmetry_rms_acc"] = float(abs(l_rms - r_rms) / (l_rms + r_rms + 1e-10))

        return f

    # ── Patient aggregation ────────────────────────────────────────────────────

    def _aggregate_to_patient(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trial-level rows to one row per patient.

        Statistics per feature (configurable): mean, std, range (max-min across
        trials), trend (OLS slope vs ordered trial index within session).

        FIX: previously fall_probability, laterality_biased, and cohort were
        included as numeric feature columns, leaking label information into the
        feature matrix. Now _META_COLS is explicitly excluded from aggregation
        so the output contains only genuine signal-derived features plus the
        bookkeeping columns needed for CV grouping and label lookup.
        """
        agg_cfg = self._patient_agg_cfg
        feat_cols = [
            c for c in trial_df.columns
            if c not in _META_COLS
            and trial_df[c].dtype in (np.float32, np.float64, np.int32, np.int64)
        ]

        patient_rows = []
        for pid, grp in trial_df.groupby("participant_id"):
            ordered = order_trial_group(grp, agg_cfg["trial_order"])
            row: dict = {
                "participant_id":    pid,
                "cohort":            grp["cohort"].iloc[0],
                "risk_label":        int(grp["risk_label"].iloc[0]),
                "multiclass_label":  int(
                    grp["multiclass_label"].iloc[0]
                    if "multiclass_label" in grp.columns
                    else grp["risk_label"].iloc[0]
                ),
                "fall_probability":  grp["fall_probability"].iloc[0],
                "laterality_biased": grp["laterality_biased"].iloc[0],
                "n_trials":          len(grp),
            }

            for col in feat_cols:
                vals = ordered[col].values.astype(float)
                stats = aggregate_trial_values(vals, agg_cfg)
                for stat_name, stat_val in stats.items():
                    row[f"{col}_{stat_name}"] = stat_val

            patient_rows.append(row)

        return pd.DataFrame(patient_rows)

    # ── IO helper ──────────────────────────────────────────────────────────────

    def _load_signals(self, trial_id: str) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}
        for pos in ["head", "lower_back", "left_foot", "right_foot"]:
            for subdir in ["signals_clean", "signals"]:
                p = self.proc_dir / subdir / f"{trial_id}_{pos}.parquet"
                if p.exists():
                    signals[pos] = pd.read_parquet(p)
                    break
        return signals