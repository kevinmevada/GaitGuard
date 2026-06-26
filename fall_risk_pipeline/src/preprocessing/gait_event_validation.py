"""
Validate algorithmic heel-strike detection against Figshare dataset annotations.
"""

from __future__ import annotations

import json
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
    resolve_trial_cohort,
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
        self.tune_cohort_percentiles = bool(val_cfg.get("tune_cohort_percentiles", False))
        self.tune_percentile_grid = [
            float(x) for x in val_cfg.get("tune_percentile_grid", [70, 75, 80, 85, 90, 95])
        ]
        self.fs = float(config["dataset"]["sampling_rate"])
        self.tolerance_samples = int(round(self.tolerance_ms * self.fs / 1000.0))

        self.processor = SignalProcessor(config)

    def run(self) -> pd.DataFrame:
        trial_dirs = self._discover_trial_dirs()
        rows: list[dict] = []
        tune_cases: list[dict] = []

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
            cohort = resolve_trial_cohort(trial_dir, meta, raw_dir=self.raw_dir)

            for side, foot_key in (("left", "left_foot"), ("right", "right_foot")):
                foot_df = self._load_foot_signal(trial_dir, foot_key)
                if foot_df is None or foot_df.empty:
                    continue

                n_samples = len(foot_df)
                gt_hs = heel_strike_indices(gt_events, side=side, n_samples=n_samples)
                if len(gt_hs) == 0:
                    continue

                detected_hs = self.processor.detect_heel_strike_indices(
                    foot_df, side, cohort=cohort
                )
                metrics = match_heel_strikes(
                    detected_hs, gt_hs, self.tolerance_samples
                )

                rows.append({
                    "trial_id": trial_id,
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "side": side,
                    "threshold_mode": self.processor.hs_threshold_mode,
                    "peak_percentile": self.processor._peak_percentile_for_cohort(cohort),
                    "tolerance_ms": self.tolerance_ms,
                    "tolerance_samples": self.tolerance_samples,
                    "gt_source": "gait_events.csv"
                    if (trial_dir / "gait_events.csv").exists()
                    else "meta_json",
                    **metrics,
                })
                tune_cases.append({
                    "cohort": cohort,
                    "side": side,
                    "foot_df": foot_df,
                    "gt_hs": gt_hs,
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

        cohort_summary_df = self._summarize_by_cohort(detail_df)
        cohort_summary_path = self.metrics_dir / "gait_event_validation_by_cohort.csv"
        cohort_summary_df.to_csv(cohort_summary_path, index=False)

        if self.tune_cohort_percentiles and tune_cases:
            tuned = self._tune_cohort_percentiles(tune_cases)
            tuned_path = self.metrics_dir / "gait_event_cohort_thresholds.json"
            tuned_path.write_text(json.dumps(tuned, indent=2, sort_keys=True), encoding="utf-8")
            logger.info(f"Cohort threshold recommendations → {tuned_path}")

        self._write_report(
            summary_df,
            detail_df,
            summary_path,
            detail_path,
            cohort_summary_df=cohort_summary_df,
            cohort_summary_path=cohort_summary_path,
        )
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
        from src.ingestion.voisard_imu_parser import voisard_txt_to_imu_frame

        df, issues, _gap = voisard_txt_to_imu_frame(
            path,
            fs=float(self.fs),
            convert_gyro_to_rad=True,
        )
        if issues:
            return None
        return df

    def _summarize_by_cohort(self, detail_df: pd.DataFrame) -> pd.DataFrame:
        if detail_df.empty or "cohort" not in detail_df.columns:
            return pd.DataFrame()

        rows: list[dict] = []
        for (cohort, side), grp in detail_df.groupby(["cohort", "side"], sort=True):
            rows.append(self._metric_row(grp, cohort=cohort, side=side))
        cohort_df = pd.DataFrame(rows)

        both_rows = []
        for cohort, grp in detail_df.groupby("cohort", sort=True):
            both_rows.append(self._metric_row(grp, cohort=cohort, side="both"))
        return pd.concat([cohort_df, pd.DataFrame(both_rows)], ignore_index=True)

    def _metric_row(
        self,
        grp: pd.DataFrame,
        *,
        cohort: str,
        side: str,
    ) -> dict:
        tp = int(grp["tp"].sum())
        fp = int(grp["fp"].sum())
        fn = int(grp["fn"].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else float("nan")
        )
        gt_total = int(grp["n_ground_truth"].sum())
        det_total = int(grp["n_detected"].sum())
        return {
            "cohort": cohort,
            "side": side,
            "n_trials": int(grp["trial_id"].nunique()),
            "n_ground_truth_hs": gt_total,
            "n_detected_hs": det_total,
            "detection_rate": float(det_total / gt_total) if gt_total > 0 else float("nan"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_abs_error_ms": float(
                grp["mean_abs_error_samples"].mean() * 1000.0 / self.fs
            ),
            "tolerance_ms": self.tolerance_ms,
        }

    def _tune_cohort_percentiles(self, tune_cases: list[dict]) -> dict[str, dict]:
        """Grid-search percentile per cohort against ground-truth heel strikes."""
        by_cohort: dict[str, list[dict]] = {}
        for case in tune_cases:
            by_cohort.setdefault(case["cohort"], []).append(case)

        recommendations: dict[str, dict] = {}
        for cohort, cases in sorted(by_cohort.items()):
            best_pct = self.processor.hs_peak_percentile
            best_f1 = -1.0
            for pct in self.tune_percentile_grid:
                scores = []
                for case in cases:
                    detected = self.processor.detect_heel_strike_indices(
                        case["foot_df"],
                        case["side"],
                        cohort=cohort,
                        peak_percentile=pct,
                    )
                    metrics = match_heel_strikes(
                        detected, case["gt_hs"], self.tolerance_samples
                    )
                    f1 = metrics["f1"]
                    if np.isfinite(f1):
                        scores.append(float(f1))
                if scores and float(np.mean(scores)) > best_f1:
                    best_f1 = float(np.mean(scores))
                    best_pct = float(pct)

            recommendations[cohort] = {
                "heel_strike_peak_percentile": best_pct,
                "mean_f1": best_f1 if best_f1 >= 0 else float("nan"),
                "threshold_mode": self.processor.hs_threshold_mode,
                "n_validation_cases": len(cases),
            }
        return recommendations

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
        *,
        cohort_summary_df: pd.DataFrame | None = None,
        cohort_summary_path: Path | None = None,
    ) -> None:
        lines = [
            "# Gait event detection validation",
            "",
            (
                "Algorithm: peak detection on `-acc_z` "
                f"({self.processor.hs_threshold_mode} mode, "
                f"{self.processor.hs_peak_percentile:.0f}th-percentile, "
                f"{self.processor.hs_min_interval_s:.1f} s minimum spacing)."
            ),
            (
                "Prominence mode ranks peaks by local prominence within each trial "
                "to reduce amplitude bias across pathologies; optional per-cohort "
                "percentiles via `heel_strike_peak_percentile_by_cohort`."
            ),
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
        if cohort_summary_df is not None and not cohort_summary_df.empty:
            lines.extend([
                "",
                "## Per-cohort detection rates",
                "",
                "| Cohort | Side | Trials | Detected/GT | Rate | F1 |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for row in cohort_summary_df.itertuples(index=False):
                if row.side != "both":
                    continue
                lines.append(
                    f"| {row.cohort} | {row.side} | {int(row.n_trials)} | "
                    f"{int(row.n_detected_hs)}/{int(row.n_ground_truth_hs)} | "
                    f"{row.detection_rate:.3f} | {row.f1:.3f} |"
                )
        lines.extend([
            "",
            f"Per-trial metrics: `{detail_path.name}`",
            f"Aggregate metrics: `{summary_path.name}`",
        ])
        if cohort_summary_path is not None:
            lines.append(f"Per-cohort metrics: `{cohort_summary_path.name}`")
        if self.tune_cohort_percentiles:
            lines.append(
                "Cohort percentile recommendations: `gait_event_cohort_thresholds.json` "
                "(copy into `heel_strike_peak_percentile_by_cohort` if systematic bias remains)."
            )
        lines.append("")
        report_path = self.metrics_dir / "gait_event_validation_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
