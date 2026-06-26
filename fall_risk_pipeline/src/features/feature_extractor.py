"""
src/features/feature_extractor.py
===================================
Stage 4: Extract temporal gait-cycle, spectral, trunk-dynamics, orientation,
nonlinear dynamics, and asymmetry features from preprocessed IMU signals.

Absolute spatial metrics use swing-phase IMU double integration (Phase 1);
see docs/spatial_features.md for calibration caveats.
Patient rows aggregate each trial feature with mean, std, range, and
session-ordered linear trend across trials.

Output:
  data/features/trial_features.parquet    — one row per trial
  data/features/patient_features.parquet  — aggregated to patient level

LEAKAGE POLICY
--------------
Label-derived target proxies (fall_probability, laterality_biased) live only in
``trial_metadata.csv`` and are never written to feature parquets. Bookkeeping
columns cohort, risk_label, and multiclass_label remain in patient parquets for
CV grouping and target lookup but are excluded from model features via
``NON_FEATURE_COLS`` / ``get_numeric_feature_columns``.
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
try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False

# Import the authoritative set of label-derived columns from data_loader so
# there is a single source of truth across the whole pipeline.
from src.dataset.label_policy import multiclass_label_from_cohort
from src.ingestion.data_loader import METADATA_ONLY_COLS
from src.features.nonlinear_metrics import (
    approximate_entropy,
    dfa_alpha,
    largest_lyapunov_exponent,
    sample_entropy,
    write_nonlinear_nan_report,
)
from src.features.feature_matrix import (
    assert_no_target_proxies_in_feature_frame,
    drop_target_proxies_from_feature_frame,
)
from src.ingestion.daphnet_label_mapping import assert_labels_not_in_feature_columns
from src.features.phase1_spatiotemporal import extract_phase1_spatiotemporal_features
from src.features.phase2_kinematic_frequency import extract_phase2_kinematic_frequency_features
from src.features.spectral_utils import harmonic_ratio_even_odd, spectral_centroid_hz, psd_band_power
from src.features.patient_temporal_aggregation import (
    aggregate_trial_values,
    order_trial_group,
    parse_patient_aggregation_config,
)

# All columns that are metadata / bookkeeping and must never be model features.
_META_COLS = METADATA_ONLY_COLS | {
    "trial_id", "participant_id", "n_trials", "session",
    "uturn_start", "uturn_end",
}


class FeatureExtractor:

    def __init__(self, config: dict):
        self.config   = config
        paths = config.get("paths", {})
        self.proc_dir = Path(paths["processed_data"])
        self.out_dir = Path(paths["features"])
        self.metrics_dir = Path(paths.get("metrics", "results/metrics"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fs = config["dataset"]["sampling_rate"]
        pp = config.get("preprocessing", {})
        self.lp_cut = float(pp.get("lowpass_cutoff_hz", 15.0))
        feat_cfg = config.get("features", {})
        self._lyap_cfg = feat_cfg.get("lyapunov", {})
        self._apen_cfg = feat_cfg.get("approximate_entropy", {})
        self._dfa_cfg  = feat_cfg.get("dfa", {})
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
        trial_df = drop_target_proxies_from_feature_frame(trial_df)
        assert_no_target_proxies_in_feature_frame(trial_df, context="trial_features.parquet")
        assert_labels_not_in_feature_columns(list(trial_df.columns), context="trial_features")

        trial_path = self.out_dir / "trial_features.parquet"
        trial_df.to_parquet(trial_path, index=False)
        logger.info(f"Trial features saved → {trial_path}  shape={trial_df.shape}")

        write_nonlinear_nan_report(trial_df, self.metrics_dir)

        patient_df = self._aggregate_to_patient(trial_df)
        patient_df = drop_target_proxies_from_feature_frame(patient_df)
        assert_no_target_proxies_in_feature_frame(
            patient_df, context="patient_features.parquet"
        )
        patient_path = self.out_dir / "patient_features.parquet"
        patient_df.to_parquet(patient_path, index=False)
        logger.info(f"Patient features saved → {patient_path}  shape={patient_df.shape}")

    def extract_trial_features_from_processed(
        self,
        processed: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> dict:
        """
        Extract one trial's feature dict from preprocessed sensor DataFrames.

        Public entry point for deployment inference (replaces direct ``_*`` calls).
        Mirrors ``_extract_trial`` but accepts in-memory processed frames instead of
        loading from disk.
        """
        feats: dict = {
            "trial_id": metadata.get("trial_id", "uploaded_trial"),
            "session": metadata.get("session"),
            "participant_id": metadata.get("participant_id", "uploaded_participant"),
            "cohort": metadata.get("cohort"),
            "risk_label": metadata.get("risk_label", 0),
            "multiclass_label": metadata.get("multiclass_label"),
        }
        if feats["multiclass_label"] is None and feats.get("cohort") is not None:
            feats["multiclass_label"] = multiclass_label_from_cohort(str(feats["cohort"]))

        lb = processed.get("lower_back")
        if lb is not None:
            feats.update(self._trunk_dynamics(lb, prefix="lb"))
            feats.update(self._spectral_features(lb, prefix="lb"))
            feats.update(self._wavelet_features(lb, prefix="lb"))
            feats.update(self._orientation_features(lb, prefix="lb"))

        lf = processed.get("left_foot")
        rf = processed.get("right_foot")
        if lf is not None and rf is not None:
            feats.update(
                extract_phase1_spatiotemporal_features(
                    lf, rf, fs=self.fs, config=self.config
                )
            )
            feats.update(self._foot_asymmetry_features(lf, rf))

        hd = processed.get("head")
        if hd is not None:
            feats.update(self._trunk_dynamics(hd, prefix="head"))
            feats.update(self._spectral_features(hd, prefix="head"))
            feats.update(self._wavelet_features(hd, prefix="head"))
            feats.update(self._orientation_features(hd, prefix="head"))

        if lb is not None and hd is not None:
            feats.update(self._head_lb_transmission(feats))

        uturn_start = metadata.get("uturn_start")
        uturn_end = metadata.get("uturn_end")
        if uturn_start is not None and uturn_end is not None:
            tid = str(metadata.get("trial_id", ""))
            lb_full = self._load_raw_signal(tid, "lower_back") if tid else None
            if lb_full is not None:
                feats.update(
                    self._turning_features(lb_full, int(uturn_start), int(uturn_end))
                )

        feats.update(
            extract_phase2_kinematic_frequency_features(
                processed, fs=self.fs, config=self.config
            )
        )

        return feats

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
            # Metadata-only — excluded from numeric aggregation and target proxies
            # (fall_probability, laterality_biased) stay in trial_metadata.csv only.
            "cohort":            row["cohort"],
            "risk_label":        row["risk_label"],
            "multiclass_label":  row.get(
                "multiclass_label",
                multiclass_label_from_cohort(str(row["cohort"])),
            ),
        }

        lb = signals.get("lower_back")
        if lb is not None:
            feats.update(self._trunk_dynamics(lb, prefix="lb"))
            feats.update(self._spectral_features(lb, prefix="lb"))
            feats.update(self._wavelet_features(lb, prefix="lb"))
            feats.update(self._orientation_features(lb, prefix="lb"))

        lf = signals.get("left_foot")
        rf = signals.get("right_foot")
        if lf is not None and rf is not None:
            feats.update(
                extract_phase1_spatiotemporal_features(
                    lf, rf, fs=self.fs, config=self.config
                )
            )
            feats.update(self._foot_asymmetry_features(lf, rf))

        hd = signals.get("head")
        if hd is not None:
            feats.update(self._trunk_dynamics(hd, prefix="head"))
            # Keep spectral/wavelet feature families symmetric with lower_back so
            # head-only and mixed-sensor ablations are not under-parameterized.
            feats.update(self._spectral_features(hd, prefix="head"))
            feats.update(self._wavelet_features(hd, prefix="head"))
            feats.update(self._orientation_features(hd, prefix="head"))

        if lb is not None and hd is not None:
            feats.update(self._head_lb_transmission(feats))

        uturn_start = row.get("uturn_start")
        uturn_end = row.get("uturn_end")
        if uturn_start is not None and uturn_end is not None:
            lb_full = self._load_raw_signal(trial_id, "lower_back")
            if lb_full is not None:
                feats.update(
                    self._turning_features(lb_full, int(uturn_start), int(uturn_end))
                )

        feats.update(
            extract_phase2_kinematic_frequency_features(
                signals, fs=self.fs, config=self.config
            )
        )

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
            f[f"{prefix}_sampen"] = sample_entropy(res, self._apen_cfg)
            f[f"{prefix}_dfa"] = dfa_alpha(res, self._dfa_cfg)

        return f

    @staticmethod
    def _head_lb_transmission(feats: dict) -> dict:
        """Head-to-lower-back postural transmission ratios.

        head_lb_rms_ratio quantifies how much acceleration propagates from the
        pelvis to the cranium during gait — a proxy for postural shock
        absorption and trunk rigidity.  head_lb_lyapunov_ratio captures the
        relative local dynamic stability between the two sites.
        """
        eps = 1e-10
        out: dict = {}
        lb_rms = feats.get("lb_rms_total")
        head_rms = feats.get("head_rms_total")
        if lb_rms is not None and head_rms is not None:
            out["head_lb_rms_ratio"] = float(head_rms / (lb_rms + eps))

        lb_lyap = feats.get("lb_lyapunov")
        head_lyap = feats.get("head_lyapunov")
        if lb_lyap is not None and head_lyap is not None:
            out["head_lb_lyapunov_ratio"] = float(head_lyap / (lb_lyap + eps))

        lb_dfa = feats.get("lb_dfa")
        head_dfa = feats.get("head_dfa")
        if lb_dfa is not None and head_dfa is not None:
            out["head_lb_dfa_ratio"] = float(head_dfa / (lb_dfa + eps))

        return out

    def _orientation_features(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        """Trunk tilt and postural stability from Madgwick quaternion fusion."""
        if "tilt_rad" not in df.columns:
            if not getattr(self, "_orientation_warned", False):
                logger.warning(
                    "Orientation columns (tilt_rad, pitch_rad, roll_rad) missing from "
                    f"cleaned signals ({prefix}). Likely cause: ahrs package not installed "
                    "or Madgwick disabled. Orientation features will be NaN for all trials."
                )
                self._orientation_warned = True
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
            return psd_band_power(freqs, pxx, f_lo, f_hi)

        f[f"{prefix}_power_0_1hz"]  = band_power(0.0, 1.0)
        f[f"{prefix}_power_1_3hz"]  = band_power(1.0, 3.0)
        f[f"{prefix}_power_3_10hz"] = band_power(3.0, 10.0)

        # Harmonic ratio: even / odd harmonic power around the dominant peak.
        # Only sum harmonics below ~80% of the preprocessing low-pass cutoff;
        # upper harmonics are attenuated by the 15 Hz filter and would otherwise
        # bias the ratio for fast-cadence gait (pathology vs speed confound).
        ratio = harmonic_ratio_even_odd(freqs, pxx, dominant_freq, self.lp_cut)
        if ratio is not None:
            f[f"{prefix}_harmonic_ratio"] = ratio

        return f

    def _wavelet_features(self, df: pd.DataFrame, prefix: str = "lb") -> dict:
        """Discrete wavelet transform energy features.

        Decomposes lower-back vertical acceleration into wavelet sub-bands
        using a Daubechies-4 wavelet (db4, 4 levels).  At 100 Hz the bands
        approximate: D1 25–50 Hz, D2 12.5–25 Hz, D3 6.25–12.5 Hz,
        D4 3.125–6.25 Hz, A4 0–3.125 Hz.  Energy ratios across bands capture
        gait regularity and pathological frequency shifts.
        """
        if not _PYWT_AVAILABLE:
            return {}

        sig_col = (
            "acc_z_grav_free" if "acc_z_grav_free" in df.columns
            else "acc_z" if "acc_z" in df.columns
            else None
        )
        if sig_col is None:
            return {}

        sig = df[sig_col].values.astype(float)
        if len(sig) < 32:
            return {}

        wavelet = "db4"
        max_level = min(4, pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len))
        if max_level < 1:
            return {}

        coeffs = pywt.wavedec(sig, wavelet, level=max_level)
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total_energy = sum(energies) + 1e-10

        f: dict = {}
        f[f"{prefix}_wavelet_energy_a{max_level}"] = energies[0]
        for i in range(1, len(energies)):
            f[f"{prefix}_wavelet_energy_d{max_level - i + 1}"] = energies[i]

        for i, e in enumerate(energies):
            level_name = f"a{max_level}" if i == 0 else f"d{max_level - i + 1}"
            f[f"{prefix}_wavelet_ratio_{level_name}"] = float(e / total_energy)

        f[f"{prefix}_wavelet_entropy"] = float(
            -sum((e / total_energy) * np.log2(e / total_energy + 1e-12)
                 for e in energies)
        )

        return f

    def _turning_features(
        self, lb: pd.DataFrame, uturn_start: int, uturn_end: int,
    ) -> dict:
        """U-turn features from the lower-back gyroscope during the annotated turn segment.

        The Figshare dataset protocol is 8.5 m walk → 180° U-turn → 8.5 m walk.
        uturnBoundaries (sample indices) mark the turn segment.  Features:
          - turn_duration_mean:   duration in seconds
          - turn_velocity_mean:   mean angular velocity (°/s) during the turn
          - turn_velocity_peak:   peak angular velocity during the turn
          - turn_jerk_mean:       mean angular jerk (rate of change of ω)
        """
        if uturn_end <= uturn_start or uturn_end > len(lb):
            return {}

        turn = lb.iloc[uturn_start:uturn_end]

        duration_s = float(len(turn)) / self.fs
        f: dict = {"turn_duration_mean": duration_s}

        gyr_col = "gyr_z" if "gyr_z" in turn.columns else None
        if gyr_col is None:
            return f

        omega = turn[gyr_col].values.astype(float)
        omega_abs = np.abs(omega)
        f["turn_velocity_mean"] = float(np.mean(omega_abs))
        f["turn_velocity_peak"] = float(np.max(omega_abs))

        angular_jerk = np.diff(omega) * self.fs
        f["turn_jerk_mean"] = float(np.mean(np.abs(angular_jerk))) if len(angular_jerk) > 0 else float("nan")

        return f

    def _foot_asymmetry_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        """Bilateral RMS acceleration asymmetry (supplements Phase 1 SI metrics)."""
        f: dict = {}
        if "acc_resultant" in lf.columns and "acc_resultant" in rf.columns:
            n = min(len(lf), len(rf))
            l_rms = float(np.sqrt(np.mean(lf["acc_resultant"].values[:n] ** 2)))
            r_rms = float(np.sqrt(np.mean(rf["acc_resultant"].values[:n] ** 2)))
            f["asymmetry_rms_acc"] = float(abs(l_rms - r_rms) / (l_rms + r_rms + 1e-10))
            f["si_rms_acc"] = float(
                abs(l_rms - r_rms) / (0.5 * (l_rms + r_rms) + 1e-10) * 100.0
            )
        return f

    # Legacy wrappers — Phase 1 module is canonical (Moon 2020 / Trabassi 2022).
    def _gait_cycle_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        return extract_phase1_spatiotemporal_features(
            lf, rf, fs=self.fs, config=self.config
        )

    def _asymmetry_features(self, lf: pd.DataFrame, rf: pd.DataFrame) -> dict:
        out = self._foot_asymmetry_features(lf, rf)
        phase1 = extract_phase1_spatiotemporal_features(
            lf, rf, fs=self.fs, config=self.config
        )
        for key in (
            "si_stride_duration",
            "si_stance_pct",
            "si_swing_pct",
            "si_step_length",
            "si_gait_velocity",
            "si_stride_duration_cv_pct",
            "stride_time_mean_asymmetry",
            "stride_time_std_asymmetry",
        ):
            if key in phase1:
                out[key] = phase1[key]
        return out

    # ── Patient aggregation ────────────────────────────────────────────────────

    def _aggregate_to_patient(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trial-level rows to one row per patient.

        Statistics per feature (configurable): mean, std, range (max-min across
        trials), trend (OLS slope vs ordered trial index within session).

        FIX: fall_probability and laterality_biased are not written to feature
        parquets (HIGH-003). _META_COLS is excluded from aggregation so the
        output contains only signal-derived features plus bookkeeping columns
        for CV grouping and label lookup.
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

    def _load_raw_signal(self, trial_id: str, position: str) -> pd.DataFrame | None:
        """Full-length ingested signal (pre U-turn exclusion) for turn-segment features."""
        path = self.proc_dir / "signals" / f"{trial_id}_{position}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)