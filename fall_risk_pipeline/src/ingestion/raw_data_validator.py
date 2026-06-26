"""
Stage 1 — Raw data validation (QC only).

Reads ``dataset_inventory.csv`` from Stage 0, runs dataset-specific checks on
Voisard IMU files and DAPHNET recordings, writes ``quality_report.csv``, and
raises ``RawDataValidationError`` when ``validation.fail_on_error`` is true.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.ingestion.dataset_discovery import (
    DAPHNET_SOURCE,
    VOISARD_SENSOR_CODES,
    VOISARD_SOURCE,
)
from src.ingestion.daphnet_parser import (
    DAPHNET_FILENAME_RE,
    DAPHNET_N_COLUMNS,
    DAPHNET_RAW_ANNOTATIONS,
)

from src.ingestion.voisard_imu_parser import (
    VOISARD_TXT_COLUMNS as VOISARD_COLUMNS,
    find_packet_counter_gaps,
    is_packet_counter_wrap,
    read_voisard_txt_raw,
)

DAPHNET_VALID_LABELS = DAPHNET_RAW_ANNOTATIONS

REPORT_COLUMNS = [
    "dataset",
    "subject",
    "cohort",
    "trial",
    "check",
    "status",
    "detail",
]


class RawDataValidationError(RuntimeError):
    """Raised when raw QC fails and ``fail_on_error`` is enabled."""


class RawDataValidator:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        processed = Path(config["paths"]["processed_data"])
        val_cfg = config.get("validation", {})
        discovery_cfg = config.get("discovery", {})

        self.inventory_path = Path(
            val_cfg.get(
                "inventory_path",
                discovery_cfg.get("inventory_path", processed / "dataset_inventory.csv"),
            )
        )
        self.report_path = Path(
            val_cfg.get("report_path", processed / "quality_report.csv")
        )
        self.fail_on_error = bool(val_cfg.get("fail_on_error", True))

        voisard_cfg = val_cfg.get("voisard", {})
        self.voisard_max_nan_fraction = float(voisard_cfg.get("max_nan_fraction", 0.0))
        self.voisard_max_packet_gap = int(voisard_cfg.get("max_packet_gap", 10))
        self.voisard_max_length_diff_rows = int(
            voisard_cfg.get("max_length_diff_rows", 25)
        )
        self.voisard_max_length_diff_fraction = float(
            voisard_cfg.get("max_length_diff_fraction", 0.02)
        )

        daphnet_cfg = val_cfg.get("daphnet", {})
        self.daphnet_max_timestamp_gap_ms = float(
            daphnet_cfg.get("max_timestamp_gap_ms", 50.0)
        )

    def run(self) -> pd.DataFrame:
        if not self.inventory_path.exists():
            raise FileNotFoundError(
                f"Dataset inventory not found: {self.inventory_path}. "
                "Run stage 'discover' first."
            )

        inventory = pd.read_csv(self.inventory_path)
        if inventory.empty:
            raise RawDataValidationError("Dataset inventory is empty.")

        logger.info(f"Raw data validation on {len(inventory)} inventoried trials")

        rows: list[dict] = []
        for record in inventory.to_dict(orient="records"):
            dataset = str(record.get("dataset", "")).lower()
            if dataset == VOISARD_SOURCE:
                rows.extend(self._validate_voisard_trial(record))
            elif dataset == DAPHNET_SOURCE:
                rows.extend(self._validate_daphnet_trial(record))
            else:
                rows.append(self._row(record, "unknown_dataset", "fail", dataset))

        report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(self.report_path, index=False)

        n_fail = int((report["status"] == "fail").sum()) if not report.empty else 0
        n_pass = int((report["status"] == "pass").sum()) if not report.empty else 0
        logger.info(
            f"Quality report → {self.report_path} "
            f"({n_pass} passed checks, {n_fail} failed checks)"
        )

        if n_fail and self.fail_on_error:
            failed_trials = (
                report.loc[report["status"] == "fail", ["dataset", "trial", "check"]]
                .drop_duplicates()
                .head(10)
            )
            preview = "; ".join(
                f"{r.dataset}/{r.trial}:{r.check}"
                for r in failed_trials.itertuples(index=False)
            )
            raise RawDataValidationError(
                f"Raw data validation failed ({n_fail} checks). "
                f"See {self.report_path}. Examples: {preview}"
            )

        return report

    def _validate_voisard_trial(self, record: dict) -> list[dict]:
        rows: list[dict] = []
        trial_dir = self.raw_dir / str(record["source_path"])
        trial_id = str(record["trial"])

        if not trial_dir.is_dir():
            return [self._row(record, "trial_path", "fail", f"missing directory {trial_dir}")]

        sensor_frames: dict[str, pd.DataFrame] = {}
        for code in VOISARD_SENSOR_CODES:
            path = trial_dir / f"{trial_id}_raw_data_{code}.txt"
            if not path.exists():
                rows.append(
                    self._row(record, f"sensor_file_{code}", "fail", f"missing {path.name}")
                )
                continue

            frame, issues = self._load_voisard_sensor(path)
            if issues:
                for issue in issues:
                    rows.append(self._row(record, f"corrupted_rows_{code}", "fail", issue))
                continue

            sensor_frames[code] = frame

            nan_frac = float(frame[VOISARD_COLUMNS[1:]].isna().mean().mean())
            if nan_frac > self.voisard_max_nan_fraction:
                rows.append(
                    self._row(
                        record,
                        f"missing_values_{code}",
                        "fail",
                        f"nan_fraction={nan_frac:.4f}",
                    )
                )
            else:
                rows.append(self._row(record, f"missing_values_{code}", "pass", ""))

            gap_issues = self._packet_counter_issues(frame["PacketCounter"])
            if gap_issues:
                rows.append(
                    self._row(
                        record,
                        f"packet_counter_{code}",
                        "fail",
                        gap_issues,
                    )
                )
            else:
                rows.append(self._row(record, f"packet_counter_{code}", "pass", ""))

        if len(sensor_frames) == len(VOISARD_SENSOR_CODES):
            lengths = {code: len(df) for code, df in sensor_frames.items()}
            values = list(lengths.values())
            spread = max(values) - min(values)
            median_len = float(np.median(values))
            allowed = max(
                self.voisard_max_length_diff_rows,
                self.voisard_max_length_diff_fraction * median_len,
            )
            if spread > allowed:
                rows.append(
                    self._row(
                        record,
                        "sensor_length_consistency",
                        "fail",
                        ", ".join(f"{k}={v}" for k, v in sorted(lengths.items())),
                    )
                )
            else:
                rows.append(self._row(record, "sensor_length_consistency", "pass", ""))

        return rows

    def _validate_daphnet_trial(self, record: dict) -> list[dict]:
        rows: list[dict] = []
        path = self.raw_dir / str(record["source_path"])

        if not path.is_file():
            return [self._row(record, "recording_file", "fail", f"missing file {path}")]

        if not DAPHNET_FILENAME_RE.match(path.name):
            return [self._row(record, "recording_file", "fail", f"unexpected name {path.name}")]

        try:
            raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
        except Exception as exc:
            return [self._row(record, "corrupted_rows", "fail", str(exc))]

        if raw.empty:
            return [self._row(record, "missing_rows", "fail", "file is empty")]

        if raw.shape[1] != DAPHNET_N_COLUMNS:
            rows.append(
                self._row(
                    record,
                    "sensor_dimensions",
                    "fail",
                    f"expected {DAPHNET_N_COLUMNS} columns, got {raw.shape[1]}",
                )
            )
        else:
            rows.append(self._row(record, "sensor_dimensions", "pass", ""))

        numeric = raw.apply(pd.to_numeric, errors="coerce")
        corrupted = int(numeric.isna().any(axis=1).sum())
        if corrupted:
            rows.append(
                self._row(record, "corrupted_rows", "fail", f"{corrupted} non-numeric rows")
            )
        else:
            rows.append(self._row(record, "corrupted_rows", "pass", ""))

        if raw.shape[1] >= DAPHNET_N_COLUMNS:
            labels = pd.to_numeric(raw.iloc[:, -1], errors="coerce")
            invalid_labels = sorted(set(labels.dropna().astype(int)) - DAPHNET_VALID_LABELS)
            if invalid_labels or labels.isna().any():
                rows.append(
                    self._row(
                        record,
                        "invalid_labels",
                        "fail",
                        f"invalid={invalid_labels}, nan_labels={int(labels.isna().sum())}",
                    )
                )
            else:
                rows.append(self._row(record, "invalid_labels", "pass", ""))

            timestamps = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            if timestamps.isna().any():
                rows.append(
                    self._row(
                        record,
                        "timestamp_continuity",
                        "fail",
                        f"{int(timestamps.isna().sum())} missing timestamps",
                    )
                )
            else:
                ts_issues = self._timestamp_issues(timestamps.to_numpy())
                if ts_issues:
                    rows.append(
                        self._row(record, "timestamp_continuity", "fail", ts_issues)
                    )
                else:
                    rows.append(self._row(record, "timestamp_continuity", "pass", ""))

        blank_rows = int((raw.astype(str).apply(lambda s: s.str.strip()) == "").all(axis=1).sum())
        if blank_rows:
            rows.append(
                self._row(record, "missing_rows", "fail", f"{blank_rows} blank rows")
            )
        else:
            rows.append(self._row(record, "missing_rows", "pass", ""))

        return rows

    def _load_voisard_sensor(self, path: Path) -> tuple[pd.DataFrame | None, list[str]]:
        return read_voisard_txt_raw(path)

    def _packet_counter_issues(self, counter: pd.Series) -> str:
        gaps = find_packet_counter_gaps(counter.to_numpy())
        if not gaps:
            return ""
        idx, prev, nxt, diff = gaps[0]
        if diff < 0:
            return (
                f"non-monotonic at rows {idx}->{idx + 1}: "
                f"{prev} to {nxt}"
            )
        if diff > self.voisard_max_packet_gap:
            return (
                f"gap at rows {idx}->{idx + 1}: "
                f"{prev} to {nxt} (max_gap={diff})"
            )
        return ""

    def _timestamp_issues(self, timestamps: np.ndarray) -> str:
        if len(timestamps) < 2:
            return ""
        diffs = np.diff(timestamps.astype(np.float64))
        if np.any(diffs < 0):
            idx = int(np.where(diffs < 0)[0][0])
            return f"non-monotonic at row {idx}: {timestamps[idx]} -> {timestamps[idx + 1]}"
        large_gaps = np.where(diffs > self.daphnet_max_timestamp_gap_ms)[0]
        if large_gaps.size:
            idx = int(large_gaps[0])
            return (
                f"gap at row {idx}: {timestamps[idx]} -> {timestamps[idx + 1]} ms "
                f"(limit {self.daphnet_max_timestamp_gap_ms})"
            )
        return ""

    @staticmethod
    def _is_packet_counter_wrap(previous: int, current: int) -> bool:
        return is_packet_counter_wrap(previous, current)

    @staticmethod
    def _row(record: dict, check: str, status: str, detail: str) -> dict:
        return {
            "dataset": record.get("dataset", ""),
            "subject": record.get("subject", ""),
            "cohort": record.get("cohort", ""),
            "trial": record.get("trial", ""),
            "check": check,
            "status": status,
            "detail": detail,
        }
