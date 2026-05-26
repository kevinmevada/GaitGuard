"""
Validate algorithmic heel-strike detection against Figshare dataset annotations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.ingestion.data_loader import SENSOR_FILE_MAPPING
from src.preprocessing.gait_events_gt import (
    heel_strike_indices,
    load_ground_truth_gait_events,
    load_trial_metadata_json,
    match_heel_strikes,
)
from src.preprocessing.signal_processor import SignalProcessor


class GaitEventValidator:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        pp = config.get("preprocessing", {})
        val_cfg = pp.get("gait_event_validation", {})
        self.tolerance_ms = float(val_cfg.get("tolerance_ms", 50.0))
        self.fs = float(config["dataset"]["sampling_rate"])
        self.tolerance_samples = int(round(self.tolerance_ms * self.fs / 1000.0))

        self.processor = SignalProcessor(config)

    def run(self) -> pd.DataFrame:
        trial_dirs = self._discover_trial_dirs()
        rows: list[dict] = []

        for trial_dir in tqdm(
            trial_dirs,
            desc="Validating gait events",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            gt_events = load_ground_truth_gait_events(trial_dir)
            if gt_events is None or gt_events.empty:
                continue

            meta = load_trial_metadata_json(trial_dir)
            trial_id = trial_dir.name
            participant_id = str(meta.get("participant_id", meta.get("subject", "unknown")))

            for side, foot_key in (("left", "left_foot"), ("right", "right_foot")):
                foot_df = self._load_foot_signal(trial_dir, foot_key)
                if foot_df is None or foot_df.empty:
                    continue

                n_samples = len(foot_df)
                gt_hs = heel_strike_indices(gt_events, side=side, n_samples=n_samples)
                if len(gt_hs) == 0:
                    continue

                detected_hs = self.processor.detect_heel_strike_indices(foot_df, side)
                metrics = match_heel_strikes(
                    detected_hs, gt_hs, self.tolerance_samples
                )

                rows.append({
                    "trial_id": trial_id,
                    "participant_id": participant_id,
                    "side": side,
                    "tolerance_ms": self.tolerance_ms,
                    "tolerance_samples": self.tolerance_samples,
                    "gt_source": "gait_events.csv"
                    if (trial_dir / "gait_events.csv").exists()
                    else "meta_json",
                    **metrics,
                })

        if not rows:
            logger.warning(
                "No trials with ground-truth gait events found under raw_data. "
                "Ensure the Figshare dataset includes gait_events.csv or *_meta.json."
            )
            return pd.DataFrame()

        detail_df = pd.DataFrame(rows)
        detail_path = self.metrics_dir / "gait_event_validation_by_trial.csv"
        detail_df.to_csv(detail_path, index=False)

        summary_df = self._summarize(detail_df)
        summary_path = self.metrics_dir / "gait_event_validation_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        self._write_report(summary_df, detail_df, summary_path, detail_path)
        logger.info(f"Gait event validation → {summary_path}")
        return summary_df

    def _discover_trial_dirs(self) -> list[Path]:
        trial_dirs: list[Path] = []
        if not self.raw_dir.exists():
            return trial_dirs
        for cohort_dir in self.raw_dir.iterdir():
            if not cohort_dir.is_dir():
                continue
            for group_dir in cohort_dir.iterdir():
                if not group_dir.is_dir():
                    continue
                for subject_dir in group_dir.iterdir():
                    if not subject_dir.is_dir():
                        continue
                    for trial_dir in subject_dir.iterdir():
                        if trial_dir.is_dir():
                            trial_dirs.append(trial_dir)
        return sorted(trial_dirs)

    def _load_foot_signal(self, trial_dir: Path, foot_key: str) -> pd.DataFrame | None:
        suffix = SENSOR_FILE_MAPPING[foot_key]
        txt_path = trial_dir / f"{trial_dir.name}_raw_data_{suffix}.txt"
        if txt_path.exists():
            return self._load_imu_txt(txt_path)
        csv_path = trial_dir / f"{foot_key}_raw.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    def _load_imu_txt(self, path: Path) -> pd.DataFrame | None:
        from src.ingestion.data_loader import IMU_AXES

        try:
            df = pd.read_csv(path, sep=r"\s+", header=None)
            if df.empty:
                return None
            n = len(df)
            df.insert(0, "time", np.arange(n) / self.fs)
            cols = ["time"] + IMU_AXES[: df.shape[1] - 1]
            df = df.iloc[:, : len(cols)]
            df.columns = cols
            return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        except Exception:
            return None

    def _summarize(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for side, grp in detail_df.groupby("side"):
            rows.append({
                "side": side,
                "n_trials": len(grp),
                "n_ground_truth_hs": int(grp["n_ground_truth"].sum()),
                "n_detected_hs": int(grp["n_detected"].sum()),
                "tp": int(grp["tp"].sum()),
                "fp": int(grp["fp"].sum()),
                "fn": int(grp["fn"].sum()),
                "precision": float(grp["tp"].sum() / (grp["tp"].sum() + grp["fp"].sum()))
                if (grp["tp"].sum() + grp["fp"].sum()) > 0
                else float("nan"),
                "recall": float(grp["tp"].sum() / (grp["tp"].sum() + grp["fn"].sum()))
                if (grp["tp"].sum() + grp["fn"].sum()) > 0
                else float("nan"),
                "mean_abs_error_ms": float(
                    grp["mean_abs_error_samples"].mean() * 1000.0 / self.fs
                ),
                "tolerance_ms": self.tolerance_ms,
            })
        summary = pd.DataFrame(rows)
        if len(summary) == 2:
            total = {
                "side": "both",
                "n_trials": int(detail_df["trial_id"].nunique()),
                "n_ground_truth_hs": int(detail_df["n_ground_truth"].sum()),
                "n_detected_hs": int(detail_df["n_detected"].sum()),
                "tp": int(detail_df["tp"].sum()),
                "fp": int(detail_df["fp"].sum()),
                "fn": int(detail_df["fn"].sum()),
                "precision": float(
                    detail_df["tp"].sum()
                    / (detail_df["tp"].sum() + detail_df["fp"].sum())
                )
                if (detail_df["tp"].sum() + detail_df["fp"].sum()) > 0
                else float("nan"),
                "recall": float(
                    detail_df["tp"].sum()
                    / (detail_df["tp"].sum() + detail_df["fn"].sum())
                )
                if (detail_df["tp"].sum() + detail_df["fn"].sum()) > 0
                else float("nan"),
                "mean_abs_error_ms": float(
                    detail_df["mean_abs_error_samples"].mean() * 1000.0 / self.fs
                ),
                "tolerance_ms": self.tolerance_ms,
            }
            p, r = total["precision"], total["recall"]
            total["f1"] = (
                float(2 * p * r / (p + r)) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else float("nan")
            )
            summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
        for i, row in summary.iterrows():
            p, r = row["precision"], row["recall"]
            if np.isfinite(p) and np.isfinite(r) and (p + r) > 0:
                summary.at[i, "f1"] = 2 * p * r / (p + r)
            else:
                summary.at[i, "f1"] = float("nan")
        return summary

    def _write_report(
        self,
        summary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        summary_path: Path,
        detail_path: Path,
    ) -> None:
        lines = [
            "# Gait event detection validation",
            "",
            "Algorithm: peak detection on `-acc_z` (75th-percentile height, 0.3 s minimum spacing).",
            f"Ground truth: Figshare `gait_events.csv` or `leftGaitEvents` / `rightGaitEvents` in trial metadata.",
            f"Match tolerance: ±{self.tolerance_ms:.0f} ms (±{self.tolerance_samples} samples at {self.fs:.0f} Hz).",
            "",
            f"Trials evaluated: {detail_df['trial_id'].nunique()}",
            "",
            "## Summary (heel strike)",
            "",
            "| Side | Trials | Precision | Recall | F1 | MAE (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary_df.itertuples(index=False):
            lines.append(
                f"| {row.side} | {int(row.n_trials)} | "
                f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
                f"{row.mean_abs_error_ms:.1f} |"
            )
        lines.extend([
            "",
            f"Per-trial metrics: `{detail_path.name}`",
            f"Aggregate metrics: `{summary_path.name}`",
            "",
        ])
        report_path = self.metrics_dir / "gait_event_validation_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
