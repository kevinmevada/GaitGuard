"""
src/ingestion/data_loader.py

Cohort resolution priority:
  1. meta.json  → "pathologyKey" (ground-truth label from Figshare dataset)
  2. meta.json  → "cohort"       (if manually annotated)
  3. path-token → _infer_cohort  (fallback for unlabeled directories)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.ingestion.voisard_imu_parser import (
    DEG_TO_RAD,
    GYR_CANONICAL,
    PacketGapInfo,
    voisard_txt_to_imu_frame,
)
from src.ingestion.daphnet_parser import (
    daphnet_frame_to_sensor_signals,
    ingest_summary_rows,
    load_daphnet_per_subject,
)
from src.ingestion.daphnet_label_mapping import (
    align_labels_to_resampled_length,
    fog_labels_path,
    save_fog_labels_npz,
)
from src.ingestion.daphnet_sensor_mapping import (
    SENSOR_MAPPING_MANIFEST,
    daphnet_trial_metadata_extras,
    map_daphnet_signals_to_voisard,
)
from src.preprocessing.daphnet_resample import (
    DAPHNET_SOURCE_FS_HZ,
    resample_daphnet_signals,
    run_psd_verification_batch,
)


COHORT_FALL_PROBABILITIES = {
    "Healthy": 5.2,
    "HipOA": 28.5,
    "KneeOA": 24.1,
    "ACL": 18.7,
    "PD": 67.3,
    "CVA": 54.2,
    "CIPN": 41.8,
    "RIL": 38.9,
}

COHORT_LABEL_MAP = {
    "Healthy": 0,
    "HipOA": 1,
    "KneeOA": 1,
    "ACL": 1,
    "PD": 2,
    "CVA": 2,
    "CIPN": 2,
    "RIL": 2,
}

LATERALITY_BIASED_COHORTS = {"HipOA", "CVA"}

# Figshare dataset pathologyKey → internal cohort name.
PATHOLOGY_KEY_MAP = {
    "HS":   "Healthy",
    "HOA":  "HipOA",
    "KOA":  "KneeOA",
    "ACL":  "ACL",
    "PD":   "PD",
    "CVA":  "CVA",
    "CIPN": "CIPN",
    "RIL":  "RIL",
}

# Columns that are label-derived and must never enter feature-space analysis.
METADATA_ONLY_COLS = {
    "cohort",
    "risk_label",
    "multiclass_label",
    "fall_probability",
    "laterality_biased",
}

IMU_AXES = [
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
]

SENSOR_FILE_MAPPING = {
    "head":        "HE",
    "lower_back":  "LB",
    "left_foot":   "LF",
    "right_foot":  "RF",
}


class TrialRecord:

    def __init__(
        self,
        trial_id: str,
        participant_id: str,
        cohort: str,
        session: str,
        age: Optional[float],
        sex: Optional[str],
        laterality: Optional[str],
        signals: dict[str, pd.DataFrame],
        gait_events: Optional[pd.DataFrame],
        risk_label: int,
        laterality_biased: bool,
        fall_probability: float,
        multiclass_label: int | None = None,
        uturn_start: int | None = None,
        uturn_end: int | None = None,
        source_dataset: str | None = None,
        sensor_mapping: str | None = None,
        eval_sensors: str | None = None,
        dropped_sensors: str | None = None,
    ):
        self.trial_id = trial_id
        self.participant_id = participant_id
        self.cohort = cohort
        self.session = session
        self.age = age
        self.sex = sex
        self.laterality = laterality
        self.signals = signals
        self.gait_events = gait_events
        self.multiclass_label = (
            int(multiclass_label) if multiclass_label is not None else int(risk_label)
        )
        self.risk_label = risk_label
        self.laterality_biased = laterality_biased
        self.fall_probability = fall_probability
        self.uturn_start = uturn_start
        self.uturn_end = uturn_end
        self.source_dataset = source_dataset
        self.sensor_mapping = sensor_mapping
        self.eval_sensors = eval_sensors
        self.dropped_sensors = dropped_sensors

    @property
    def duration_s(self) -> float:
        for key in ("lower_back", "trunk", "ankle", "thigh", "head", "left_foot", "right_foot"):
            ref = self.signals.get(key)
            if ref is not None and not ref.empty and "time" in ref.columns:
                return float(ref["time"].iloc[-1] - ref["time"].iloc[0])
        return 0.0

    def to_meta_dict(self) -> dict:
        return {
            "trial_id":         self.trial_id,
            "participant_id":   self.participant_id,
            "cohort":           self.cohort,
            "session":          self.session,
            "age":              self.age,
            "sex":              self.sex,
            "laterality":       self.laterality,
            "risk_label":       self.risk_label,
            "multiclass_label": self.multiclass_label,
            "laterality_biased": self.laterality_biased,
            "fall_probability": self.fall_probability,
            "duration_s":       self.duration_s,
            "has_gait_events_gt": self.gait_events is not None and not self.gait_events.empty,
            "uturn_start":      self.uturn_start,
            "uturn_end":        self.uturn_end,
            "source_dataset":   self.source_dataset,
            "sensor_mapping":   self.sensor_mapping,
            "eval_sensors":     self.eval_sensors,
            "dropped_sensors":  self.dropped_sensors,
        }


class DataLoader:

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.out_dir = Path(config["paths"]["processed_data"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.min_duration = config["preprocessing"]["min_trial_length_s"]
        from src.dataset.label_policy import get_dataset_label_config

        self._label_cfg = get_dataset_label_config(config)
        self.fs = config["dataset"]["sampling_rate"]

        ingest_cfg = config.get("ingestion", {})
        self.convert_gyro_to_rad = bool(ingest_cfg.get("convert_gyro_to_rad", True))
        self.packet_gap_strategy = str(
            ingest_cfg.get("packet_gap_strategy", "interpolate")
        ).lower()
        self.max_interpolate_gap = int(ingest_cfg.get("max_interpolate_gap", 10))
        daphnet_cfg = ingest_cfg.get("daphnet", {})
        self.daphnet_enabled = bool(daphnet_cfg.get("enabled", True))
        self.daphnet_drop_calibration = bool(daphnet_cfg.get("drop_annotation_zero", True))
        self.daphnet_source_fs = float(daphnet_cfg.get("source_fs_hz", DAPHNET_SOURCE_FS_HZ))
        self.daphnet_target_fs = float(
            daphnet_cfg.get("target_fs_hz", config["dataset"]["sampling_rate"])
        )
        self.daphnet_resample_up = int(daphnet_cfg.get("resample_up", 25))
        self.daphnet_resample_down = int(daphnet_cfg.get("resample_down", 16))
        psd_cfg = daphnet_cfg.get("psd_verification") or {}
        self.daphnet_psd_enabled = bool(psd_cfg.get("enabled", True))
        self.daphnet_psd_min_subjects = int(psd_cfg.get("min_subjects", 2))
        band = psd_cfg.get("band_hz", [3.0, 8.0])
        self.daphnet_psd_band = (float(band[0]), float(band[1]))
        self.daphnet_psd_max_shift = float(psd_cfg.get("max_peak_shift_hz", 0.5))
        fig_rel = psd_cfg.get("figure_dir", "results/figs")
        pipeline_root = Path(__file__).resolve().parents[2]
        self.daphnet_psd_figure_dir = (
            Path(fig_rel) if Path(fig_rel).is_absolute() else pipeline_root / fig_rel
        )
        self._packet_gap_rows: list[dict] = []
        self._daphnet_ingest_rows: list[dict] = []
        self._daphnet_psd_rows: list[dict] = []
        self._daphnet_fog_labels: dict[str, np.ndarray] = {}

    # ─────────────────────────────────────────

    def _find_trial_dirs(self) -> list[Path]:
        """Locate trial folders (``*_meta.json`` parent), preferring ``raw/voisard``."""
        voisard_root = self.raw_dir / "voisard"
        scan_root = voisard_root if voisard_root.is_dir() else self.raw_dir
        return sorted(
            meta_path.parent
            for meta_path in scan_root.rglob("*_meta.json")
            if meta_path.is_file()
        )

    def trial_dir_index(self) -> dict[str, Path]:
        return {td.name: td for td in self._find_trial_dirs()}

    def find_trial_dir(self, trial_id: str) -> Path | None:
        return self.trial_dir_index().get(str(trial_id))

    def _save_records_to_dir(self, records: list[TrialRecord], out_dir: Path) -> None:
        """Write parquets + gait events for records under ``out_dir`` (shard layout)."""
        signals_dir = out_dir / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            for pos, df in rec.signals.items():
                if df is None or df.empty:
                    continue
                path = signals_dir / f"{rec.trial_id}_{pos}.parquet"
                df.to_parquet(path, index=False)
            if rec.gait_events is not None and not rec.gait_events.empty:
                ge_dir = out_dir / "gait_events"
                ge_dir.mkdir(exist_ok=True)
                rec.gait_events.to_csv(ge_dir / f"{rec.trial_id}.csv", index=False)

    def run_voisard_shard(self, trial_ids: list[str], shard_out: Path) -> dict:
        """
        Ingest a manifest chunk of Voisard trials into ``shard_out``.

        Writes ``trial_metadata_chunk.csv``, ``signals/``, optional ``packet_gap_chunk.csv``.
        """
        shard_out.mkdir(parents=True, exist_ok=True)
        index = self.trial_dir_index()
        records: list[TrialRecord] = []
        skipped: list[str] = []
        self._packet_gap_rows = []

        for trial_id in trial_ids:
            td = index.get(str(trial_id))
            if td is None:
                skipped.append(str(trial_id))
                logger.warning("Shard ingest: missing trial dir for {}", trial_id)
                continue
            try:
                rec = self._load_trial(td)
                if rec.duration_s < self.min_duration:
                    skipped.append(str(trial_id))
                    continue
                records.append(rec)
            except Exception as exc:
                skipped.append(str(trial_id))
                logger.warning("Shard ingest: skipping {}: {}", trial_id, exc)

        if records:
            meta_df = pd.DataFrame([r.to_meta_dict() for r in records])
            meta_df.to_csv(shard_out / "trial_metadata_chunk.csv", index=False)
            self._save_records_to_dir(records, shard_out)

        if self._packet_gap_rows:
            pd.DataFrame(self._packet_gap_rows).to_csv(
                shard_out / "packet_gap_chunk.csv", index=False
            )

        logger.info(
            "Shard ingest → {} ({} loaded, {} skipped)",
            shard_out,
            len(records),
            len(skipped),
        )
        return {"loaded": len(records), "skipped": skipped, "shard_out": str(shard_out)}

    def run_daphnet_ingest(self) -> list[TrialRecord]:
        """Ingest DAPHNET only (run once after Voisard shard merge)."""
        records: list[TrialRecord] = []
        added, skipped = self._ingest_daphnet_subjects(records)
        logger.info("DAPHNET ingest: {} added, {} skipped", added, skipped)
        self._write_daphnet_ingest_report()
        return records

    def append_daphnet_to_processed(self, daphnet_records: list[TrialRecord]) -> None:
        """Append DAPHNET trials to existing ``trial_metadata.csv`` and save signals."""
        if not daphnet_records:
            return
        meta_path = self.out_dir / "trial_metadata.csv"
        existing = pd.read_csv(meta_path) if meta_path.is_file() else pd.DataFrame()
        new_meta = pd.DataFrame([r.to_meta_dict() for r in daphnet_records])
        pd.concat([existing, new_meta], ignore_index=True).to_csv(meta_path, index=False)
        self._save_records_to_dir(daphnet_records, self.out_dir)
        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if self._daphnet_fog_labels:
            trial_ids = {sid: f"daphnet_{sid}" for sid in self._daphnet_fog_labels}
            save_fog_labels_npz(fog_labels_path(self.config), self._daphnet_fog_labels, trial_ids)
        try:
            from src.reporting.demographics_table import generate_demographics_table

            generate_demographics_table(self.config)
        except Exception as exc:
            logger.warning(f"Demographics table skipped: {exc}")

    def run(self) -> list[TrialRecord]:
        logger.info(f"Scanning raw data: {self.raw_dir}")

        trial_dirs = self._find_trial_dirs()

        records: list[TrialRecord] = []
        skipped = 0

        if not trial_dirs:
            logger.warning("No Voisard trials found.")
        else:
            for td in tqdm(
                trial_dirs,
                desc="Loading trials",
                colour="red",
                bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
            ):
                try:
                    rec = self._load_trial(td)

                    if rec.duration_s < self.min_duration:
                        skipped += 1
                        continue

                    records.append(rec)

                except Exception as e:
                    logger.warning(f"Skipping {td.name}: {e}")
                    skipped += 1

            logger.info(f"Loaded {len(records)} Voisard trials ({skipped} skipped)")

        daphnet_added, daphnet_skipped = self._ingest_daphnet_subjects(records)
        if daphnet_added:
            logger.info(
                f"Ingested {daphnet_added} DAPHNET subject bundles "
                f"({daphnet_skipped} skipped)"
            )

        self._write_packet_gap_report(len(trial_dirs))
        self._write_daphnet_ingest_report()
        if not records:
            return []
        self._save(records)

        return records

    # ─────────────────────────────────────────

    def _infer_cohort(self, trial_dir: Path) -> str:
        """
        FIX: tokenise the path on separators and underscores, then match whole
        tokens (case-insensitive) rather than arbitrary substrings.

        Previously `"hs" in path` matched any path containing "hs" anywhere
        (e.g. "thresholds", "mesh_data"), silently assigning the Healthy label.
        Whole-token matching eliminates those false positives.
        """
        # Build a set of lowercase tokens from every path component.
        tokens: set[str] = set()
        for part in trial_dir.parts:
            part_lower = part.lower()
            tokens.add(part_lower)
            # Split on underscores, hyphens, and periods so "pd-001" → {"pd", "001"}.
            tokens.update(t for t in re.split(r"[_\-.]", part_lower) if t)

        # Ordered from most specific to most general to avoid mis-classification.
        if "healthy" in tokens or "hs" in tokens:
            return "Healthy"
        if "hipoa" in tokens or "hoa" in tokens:
            return "HipOA"
        if "kneeoa" in tokens or "koa" in tokens:
            return "KneeOA"
        if "cipn" in tokens:
            return "CIPN"
        if "acl" in tokens:
            return "ACL"
        if "cva" in tokens:
            return "CVA"
        if "ril" in tokens:
            return "RIL"
        if "pd" in tokens:
            return "PD"

        return "Unknown"

    # ─────────────────────────────────────────

    def _load_trial(self, trial_dir: Path) -> TrialRecord:
        meta = self._load_metadata(trial_dir)

        if "cohort" not in meta:
            pkey = meta.get("pathologyKey", "")
            if pkey in PATHOLOGY_KEY_MAP:
                meta["cohort"] = PATHOLOGY_KEY_MAP[pkey]
            else:
                meta["cohort"] = self._infer_cohort(trial_dir)

        cohort    = meta.get("cohort", "Unknown")
        if cohort == "Unknown":
            logger.warning(
                f"Cohort UNKNOWN for {trial_dir.name} "
                f"(pathologyKey={meta.get('pathologyKey', 'MISSING')}). "
                f"Check PATHOLOGY_KEY_MAP or _infer_cohort."
            )
        from src.dataset.label_policy import resolve_labels

        resolved = resolve_labels(cohort, self.config)
        risk_label = resolved.training_label
        fall_probability = resolved.fall_probability
        multiclass_label = resolved.multiclass_label

        signals: dict[str, pd.DataFrame] = {}

        for pos, suffix in SENSOR_FILE_MAPPING.items():
            txt_path = trial_dir / f"{trial_dir.name}_raw_data_{suffix}.txt"
            df = None

            declared_duration = meta.get("duration_seconds")
            declared_duration = (
                float(declared_duration)
                if declared_duration is not None
                else None
            )

            if txt_path.exists():
                df, gap_info = self._load_imu_txt(txt_path)
                if gap_info.n_gaps:
                    self._packet_gap_rows.append(
                        {
                            "trial_id": trial_dir.name,
                            "sensor_file": txt_path.name,
                            "sensor": suffix,
                            "n_gaps": gap_info.n_gaps,
                            "max_gap": gap_info.max_gap,
                            "gap_indices": ";".join(str(i) for i in gap_info.gap_indices),
                            "gap_details": "; ".join(gap_info.gap_details),
                            "repaired": gap_info.repaired,
                            "truncated": gap_info.truncated,
                            "n_rows_before": gap_info.n_rows_before,
                            "n_rows_after": gap_info.n_rows_after,
                        }
                    )

            if df is None:
                csv_path = trial_dir / f"{pos}_raw.csv"
                if csv_path.exists():
                    df = self._load_imu_csv(
                        csv_path,
                        declared_duration_s=declared_duration,
                    )

            if df is None or df.empty or len(df) < 50:
                continue

            signals[pos] = df

        if not signals:
            raise ValueError("No valid signals")

        expected_sensors = set(SENSOR_FILE_MAPPING)
        missing_sensors = expected_sensors - set(signals)
        if missing_sensors:
            logger.warning(
                f"{trial_dir.name}: missing sensors {sorted(missing_sensors)} — "
                "bilateral/asymmetry and cross-sensor features may be degraded"
            )

        from src.preprocessing.gait_events_gt import load_ground_truth_gait_events

        gait_events = load_ground_truth_gait_events(trial_dir)

        uturn_bounds = meta.get("uturnBoundaries")
        uturn_start = int(uturn_bounds[0]) if uturn_bounds and len(uturn_bounds) >= 2 else None
        uturn_end = int(uturn_bounds[1]) if uturn_bounds and len(uturn_bounds) >= 2 else None

        return TrialRecord(
            trial_id=trial_dir.name,
            participant_id=str(meta.get("participant_id", "unknown")),
            cohort=cohort,
            session=str(meta.get("session", "1")),
            age=meta.get("age"),
            sex=meta.get("sex"),
            laterality=meta.get("laterality"),
            signals=signals,
            gait_events=gait_events,
            risk_label=risk_label,
            multiclass_label=multiclass_label,
            laterality_biased=cohort in LATERALITY_BIASED_COHORTS,
            fall_probability=fall_probability,
            uturn_start=uturn_start,
            uturn_end=uturn_end,
        )

    # ─────────────────────────────────────────

    def _load_metadata(self, trial_dir: Path) -> dict:
        meta_path = trial_dir / f"{trial_dir.name}_meta.json"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

            if "subject" in meta:
                meta["participant_id"] = meta["subject"]

            if "gender" in meta:
                meta["sex"] = meta["gender"]

            return meta

        return {}

    # ─────────────────────────────────────────

    def _load_imu_txt(self, path: Path) -> tuple[Optional[pd.DataFrame], PacketGapInfo]:
        try:
            df, issues, gap_info = voisard_txt_to_imu_frame(
                path,
                fs=float(self.fs),
                convert_gyro_to_rad=self.convert_gyro_to_rad,
                gap_strategy=(
                    "interpolate"
                    if self.packet_gap_strategy == "interpolate"
                    else "truncate"
                ),
                max_interpolate_gap=self.max_interpolate_gap,
            )
            if issues:
                logger.warning(f"Failed loading {path}: {'; '.join(issues)}")
                return None, gap_info
            return df, gap_info
        except Exception as exc:
            logger.warning(f"Failed loading {path}: {exc}")
            return None, PacketGapInfo(sensor_path=str(path))

    # ─────────────────────────────────────────

    def _load_imu_csv(
        self, path: Path, declared_duration_s: float | None = None
    ) -> pd.DataFrame:
        df = pd.read_csv(path)

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "time" not in df.columns:
            n = len(df)
            if (
                declared_duration_s is not None
                and np.isfinite(declared_duration_s)
                and declared_duration_s > 0
                and n > 1
            ):
                # Keep fixture metadata duration consistent with generated time.
                df.insert(0, "time", np.linspace(0.0, float(declared_duration_s), n))
            else:
                df.insert(0, "time", df.index / self.fs)

        keep = ["time"] + [c for c in IMU_AXES if c in df.columns]

        df = df[keep].apply(pd.to_numeric, errors="coerce").dropna()

        if self.convert_gyro_to_rad:
            for col in GYR_CANONICAL:
                if col in df.columns:
                    df[col] = df[col].astype(float) * DEG_TO_RAD

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────

    def _write_packet_gap_report(self, n_trials_scanned: int) -> None:
        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)

        gap_df = pd.DataFrame(self._packet_gap_rows)
        gap_path = metrics_dir / "packet_gap_report.csv"
        gap_df.to_csv(gap_path, index=False)

        trials_with_gaps = (
            gap_df["trial_id"].nunique() if not gap_df.empty and "trial_id" in gap_df.columns else 0
        )
        frac = trials_with_gaps / max(n_trials_scanned, 1)
        summary_lines = [
            "# packet_gap_summary — cite in dataset section if trials_with_gaps_pct > 5%",
            f"trials_scanned: {n_trials_scanned}",
            f"trials_with_packet_gaps: {trials_with_gaps}",
            f"trials_with_gaps_pct: {frac * 100:.2f}",
            f"sensor_files_with_gaps: {len(gap_df)}",
            f"report_csv: {gap_path.name}",
        ]
        if frac > 0.05:
            summary_lines.append(
                "reviewer_note: >5% of trials had PacketCounter gaps; document in Methods/Dataset."
            )
        (metrics_dir / "packet_gap_summary.log").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )
        if trials_with_gaps:
            logger.warning(
                "PacketCounter gaps in {}/{} trials ({:.1f}%) — see {}",
                trials_with_gaps,
                n_trials_scanned,
                frac * 100,
                gap_path,
            )
        else:
            logger.info("No PacketCounter gaps detected across ingested Voisard txt files.")

    def _ingest_daphnet_subjects(self, records: list[TrialRecord]) -> tuple[int, int]:
        """Parse flat DAPHNET files; concatenate recordings per subject."""
        if not self.daphnet_enabled:
            return 0, 0

        root = self.raw_dir / "daphnet"
        if not root.is_dir():
            return 0, 0

        bundles = load_daphnet_per_subject(
            root,
            drop_calibration=self.daphnet_drop_calibration,
        )
        if not bundles:
            return 0, 0

        self._daphnet_ingest_rows = ingest_summary_rows(bundles)
        added = 0
        skipped = 0
        psd_queue: list[tuple[str, np.ndarray, np.ndarray]] = []
        pending_records: list[tuple[str, dict[str, pd.DataFrame], object]] = []

        for subject_id, bundle in bundles.items():
            if bundle.n_rows == 0:
                skipped += 1
                continue

            signals_64 = daphnet_frame_to_sensor_signals(bundle.frame)
            trunk_64 = signals_64.get("trunk")
            if trunk_64 is None or trunk_64.empty:
                skipped += 1
                continue

            duration_s = float(trunk_64["time"].iloc[-1] - trunk_64["time"].iloc[0])
            if duration_s < self.min_duration:
                skipped += 1
                logger.warning(
                    f"Skipping DAPHNET {subject_id}: duration {duration_s:.1f}s "
                    f"< min {self.min_duration}s"
                )
                continue

            trunk_z_before = trunk_64["acc_z"].to_numpy()
            trunk_100 = resample_daphnet_signals(
                {"trunk": trunk_64},
                up=self.daphnet_resample_up,
                down=self.daphnet_resample_down,
                target_fs_hz=self.daphnet_target_fs,
            )
            trunk_z_after = trunk_100["trunk"]["acc_z"].to_numpy()
            n_resampled = len(trunk_z_after)
            y_true = align_labels_to_resampled_length(
                bundle.frame["annotation"].to_numpy(),
                n_resampled,
            )
            psd_queue.append((subject_id, trunk_z_before, trunk_z_after))
            voisard_signals = map_daphnet_signals_to_voisard(trunk_100)
            if not voisard_signals:
                skipped += 1
                continue
            self._daphnet_fog_labels[subject_id] = y_true
            pending_records.append((subject_id, voisard_signals, bundle))

        if pending_records and self.daphnet_psd_enabled:
            self._daphnet_psd_rows = [
                r.to_dict()
                for r in run_psd_verification_batch(
                    psd_queue,
                    figure_dir=self.daphnet_psd_figure_dir,
                    fs_before=self.daphnet_source_fs,
                    fs_after=self.daphnet_target_fs,
                    band_hz=self.daphnet_psd_band,
                    max_peak_shift_hz=self.daphnet_psd_max_shift,
                    min_subjects=self.daphnet_psd_min_subjects,
                )
            ]

        from src.dataset.label_policy import resolve_labels

        resolved = resolve_labels("PD", self.config)
        mapping_meta = daphnet_trial_metadata_extras()

        for subject_id, signals, bundle in pending_records:
            trial_id = f"daphnet_{subject_id}"
            records.append(
                TrialRecord(
                    trial_id=trial_id,
                    participant_id=subject_id,
                    cohort="PD",
                    session="+".join(bundle.source_trials),
                    age=None,
                    sex=None,
                    laterality=None,
                    signals=signals,
                    gait_events=None,
                    risk_label=resolved.training_label,
                    multiclass_label=resolved.multiclass_label,
                    laterality_biased=False,
                    fall_probability=resolved.fall_probability,
                    source_dataset=mapping_meta["source_dataset"],
                    sensor_mapping=mapping_meta["sensor_mapping"],
                    eval_sensors=mapping_meta["eval_sensors"],
                    dropped_sensors=mapping_meta["dropped_sensors"],
                )
            )
            added += 1

        self._write_daphnet_fog_labels()

        return added, skipped

    def _write_daphnet_fog_labels(self) -> None:
        if not self._daphnet_fog_labels:
            return
        out = fog_labels_path(self.out_dir)
        trial_ids = {sid: f"daphnet_{sid}" for sid in self._daphnet_fog_labels}
        save_fog_labels_npz(self._daphnet_fog_labels, out, trial_ids=trial_ids)
        logger.info(
            "DAPHNET FOG labels (eval-only, separate from features) → {} ({} subjects)",
            out,
            len(self._daphnet_fog_labels),
        )

    def _write_daphnet_ingest_report(self) -> None:
        if not self._daphnet_ingest_rows:
            return
        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / "daphnet_ingest_report.csv"
        pd.DataFrame(self._daphnet_ingest_rows).to_csv(out, index=False)
        n_files = len(self._daphnet_ingest_rows)
        n_dropped = int(
            sum(r.get("n_rows_dropped_calibration", 0) for r in self._daphnet_ingest_rows)
        )
        logger.info(
            "DAPHNET ingest: {} files, {} calibration rows dropped → {}",
            n_files,
            n_dropped,
            out,
        )
        mapping_path = metrics_dir / "daphnet_sensor_mapping.json"
        mapping_path.write_text(
            json.dumps(SENSOR_MAPPING_MANIFEST, indent=2),
            encoding="utf-8",
        )
        logger.info("DAPHNET sensor mapping manifest → {}", mapping_path)
        if self._daphnet_psd_rows:
            psd_out = metrics_dir / "daphnet_psd_verification.csv"
            pd.DataFrame(self._daphnet_psd_rows).to_csv(psd_out, index=False)
            logger.info(
                "DAPHNET PSD verification: {} subjects → {} (figs: {})",
                len(self._daphnet_psd_rows),
                psd_out,
                self.daphnet_psd_figure_dir,
            )

    def _save(self, records: list[TrialRecord]):
        meta_df = pd.DataFrame([r.to_meta_dict() for r in records])
        meta_df.to_csv(self.out_dir / "trial_metadata.csv", index=False)

        signals_dir = self.out_dir / "signals"
        signals_dir.mkdir(exist_ok=True)

        for rec in tqdm(
            records,
            desc="Saving signals",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            for pos, df in rec.signals.items():
                if df is None or df.empty:
                    continue

                path = signals_dir / f"{rec.trial_id}_{pos}.parquet"
                df.to_parquet(path, index=False)

            if rec.gait_events is not None and not rec.gait_events.empty:
                ge_dir = self.out_dir / "gait_events"
                ge_dir.mkdir(exist_ok=True)
                rec.gait_events.to_csv(ge_dir / f"{rec.trial_id}.csv", index=False)

        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.dataset.label_balance import save_class_distribution_reports

            if "participant_id" in meta_df.columns:
                participants = meta_df.drop_duplicates("participant_id")
                save_class_distribution_reports(
                    participants, metrics_dir, config=self.config
                )
            else:
                save_class_distribution_reports(meta_df, metrics_dir, config=self.config)
        except Exception as exc:
            logger.warning(f"Class distribution report skipped: {exc}")

        try:
            from src.reporting.demographics_table import generate_demographics_table

            generate_demographics_table(self.config)
        except Exception as exc:
            logger.warning(f"Demographics table skipped: {exc}")

        logger.info("Data saved successfully")