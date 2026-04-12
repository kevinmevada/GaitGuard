"""
src/ingestion/data_loader.py
FINAL FIXED VERSION (robust + safe)

FIX: _infer_cohort now matches on full path-segment tokens (split on os.sep and
     underscore) instead of raw substring search. This prevents false positives
     such as "thresholds" matching "hs" (Healthy), or "mesh" matching "hs".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


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

# Columns that are label-derived and must never enter feature-space analysis.
METADATA_ONLY_COLS = {
    "cohort",
    "risk_label",
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
        self.risk_label = risk_label
        self.laterality_biased = laterality_biased
        self.fall_probability = fall_probability

    @property
    def duration_s(self) -> float:
        ref = self.signals.get("lower_back")
        if ref is None or "time" not in ref.columns:
            return 0.0
        return float(ref["time"].iloc[-1] - ref["time"].iloc[0])

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
            "laterality_biased": self.laterality_biased,
            "fall_probability": self.fall_probability,
            "duration_s":       self.duration_s,
        }


class DataLoader:

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.out_dir = Path(config["paths"]["processed_data"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.min_duration = config["preprocessing"]["min_trial_length_s"]
        self.label_mode   = config["dataset"]["label_mode"]
        self.threshold    = config["dataset"]["high_risk_threshold"]
        self.fs           = config["dataset"]["sampling_rate"]

    # ─────────────────────────────────────────

    def run(self) -> list[TrialRecord]:
        logger.info(f"Scanning raw data: {self.raw_dir}")

        trial_dirs = []

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

        trial_dirs = sorted(trial_dirs)

        if not trial_dirs:
            logger.warning("No trials found.")
            return []

        records = []
        skipped = 0

        for td in tqdm(trial_dirs, desc="Loading trials"):
            try:
                rec = self._load_trial(td)

                if rec.duration_s < self.min_duration:
                    skipped += 1
                    continue

                records.append(rec)

            except Exception as e:
                logger.warning(f"Skipping {td.name}: {e}")
                skipped += 1

        logger.info(f"Loaded {len(records)} trials ({skipped} skipped)")

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
            # Also split on underscores so "pd_001" yields {"pd", "001"}.
            tokens.update(part_lower.split("_"))

        # Ordered from most specific to most general to avoid mis-classification.
        if "healthy" in tokens or "hs" in tokens:
            return "Healthy"
        if "hipoa" in tokens:
            return "HipOA"
        if "kneeoa" in tokens:
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
            meta["cohort"] = self._infer_cohort(trial_dir)

        cohort    = meta.get("cohort", "Unknown")
        raw_label = COHORT_LABEL_MAP.get(cohort, 0)

        risk_label = (
            int(raw_label >= self.threshold)
            if self.label_mode == "binary"
            else raw_label
        )

        fall_probability = COHORT_FALL_PROBABILITIES.get(cohort, 10.0) / 100.0

        signals: dict[str, pd.DataFrame] = {}

        for pos, suffix in SENSOR_FILE_MAPPING.items():
            txt_path = trial_dir / f"{trial_dir.name}_raw_data_{suffix}.txt"
            df = None

            if txt_path.exists():
                df = self._load_imu_txt(txt_path)

            if df is None:
                csv_path = trial_dir / f"{pos}_raw.csv"
                if csv_path.exists():
                    df = self._load_imu_csv(csv_path)

            if df is None or df.empty or len(df) < 50:
                continue

            signals[pos] = df

        if not signals:
            raise ValueError("No valid signals")

        gait_events = None
        events_path = trial_dir / "gait_events.csv"
        if events_path.exists():
            gait_events = pd.read_csv(events_path)

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
            laterality_biased=cohort in LATERALITY_BIASED_COHORTS,
            fall_probability=fall_probability,
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

    def _load_imu_txt(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, sep=r"\s+", header=None)

            if df.empty:
                return None

            n = len(df)
            time_col = np.arange(n) / self.fs

            df.insert(0, "time", time_col)

            available_axes = df.shape[1] - 1
            cols = ["time"] + IMU_AXES[:available_axes]

            df = df.iloc[:, :len(cols)]
            df.columns = cols

            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            return df.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Failed loading {path}: {e}")
            return None

    # ─────────────────────────────────────────

    def _load_imu_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "time" not in df.columns:
            df.insert(0, "time", df.index / self.fs)

        keep = ["time"] + [c for c in IMU_AXES if c in df.columns]

        df = df[keep].apply(pd.to_numeric, errors="coerce").dropna()

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────

    def _save(self, records: list[TrialRecord]):
        meta_df = pd.DataFrame([r.to_meta_dict() for r in records])
        meta_df.to_csv(self.out_dir / "trial_metadata.csv", index=False)

        signals_dir = self.out_dir / "signals"
        signals_dir.mkdir(exist_ok=True)

        for rec in tqdm(records, desc="Saving signals"):
            for pos, df in rec.signals.items():
                if df is None or df.empty:
                    continue

                path = signals_dir / f"{rec.trial_id}_{pos}.parquet"
                df.to_parquet(path, index=False)

        logger.info("Data saved successfully")