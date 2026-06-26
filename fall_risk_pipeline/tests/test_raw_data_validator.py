"""Tests for Stage 1 raw data validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.raw_data_validator import RawDataValidationError, RawDataValidator


def _config(raw: Path, processed: Path, *, fail_on_error: bool = True) -> dict:
    inventory = processed / "dataset_inventory.csv"
    return {
        "paths": {
            "raw_data": str(raw),
            "processed_data": str(processed),
        },
        "discovery": {"inventory_path": str(inventory)},
        "validation": {
            "inventory_path": str(inventory),
            "report_path": str(processed / "quality_report.csv"),
            "fail_on_error": fail_on_error,
        },
    }


def _write_inventory(processed: Path, rows: list[dict]) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "dataset_inventory.csv", index=False)


def _write_voisard_trial(
    trial_dir: Path,
    *,
    trial_id: str = "HS_1_1",
    subject: str = "HS_1",
    rows_per_sensor: int = 5,
    lb_rows: int | None = None,
    packet_start: int = 100,
    gap_at: int | None = None,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    meta = {"subject": subject, "pathologyKey": "HS"}
    (trial_dir / f"{trial_id}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    for code in ["HE", "LB", "LF", "RF"]:
        n_rows = lb_rows if code == "LB" and lb_rows is not None else rows_per_sensor
        lines = ["PacketCounter\tAcc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z"]
        counter = packet_start
        for i in range(n_rows):
            if gap_at is not None and i == gap_at:
                counter += 5
            lines.append(
                f"{counter}\t1.0\t2.0\t3.0\t4.0\t5.0\t6.0\t7.0\t8.0\t9.0"
            )
            counter += 1
        (trial_dir / f"{trial_id}_raw_data_{code}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def test_validate_raw_passes_clean_voisard_and_daphnet(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    trial_dir = raw / "voisard" / "healthy" / "HS" / "HS_1" / "HS_1_1"
    _write_voisard_trial(trial_dir)

    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)
    (daphnet_dir / "S01R01.txt").write_text(
        "15 70 39 -970 0 0 0 0 0 0 0\n31 70 39 -970 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    _write_inventory(
        processed,
        [
            {
                "dataset": "voisard",
                "subject": "HS_1",
                "cohort": "Healthy",
                "trial": "HS_1_1",
                "source_path": "voisard/healthy/HS/HS_1/HS_1_1",
            },
            {
                "dataset": "daphnet",
                "subject": "S01",
                "cohort": "PD",
                "trial": "S01R01",
                "source_path": "daphnet/S01R01.txt",
            },
        ],
    )

    report = RawDataValidator(_config(raw, processed)).run()
    assert (report["status"] == "fail").sum() == 0
    assert (processed / "quality_report.csv").exists()


def test_validate_raw_fails_packet_counter_gap(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    trial_dir = raw / "voisard" / "healthy" / "HS" / "HS_1" / "HS_1_1"
    _write_voisard_trial(trial_dir, gap_at=2)

    _write_inventory(
        processed,
        [
            {
                "dataset": "voisard",
                "subject": "HS_1",
                "cohort": "Healthy",
                "trial": "HS_1_1",
                "source_path": "voisard/healthy/HS/HS_1/HS_1_1",
            }
        ],
    )

    cfg = _config(raw, processed)
    cfg["validation"]["voisard"] = {
        "max_packet_gap": 1,
        "max_length_diff_rows": 0,
    }

    with pytest.raises(RawDataValidationError):
        RawDataValidator(cfg).run()

    report = pd.read_csv(processed / "quality_report.csv")
    assert ((report["check"] == "packet_counter_HE") & (report["status"] == "fail")).any()


def test_validate_raw_fails_sensor_length_mismatch(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    trial_dir = raw / "voisard" / "healthy" / "HS" / "HS_1" / "HS_1_1"
    _write_voisard_trial(trial_dir, lb_rows=3)

    _write_inventory(
        processed,
        [
            {
                "dataset": "voisard",
                "subject": "HS_1",
                "cohort": "Healthy",
                "trial": "HS_1_1",
                "source_path": "voisard/healthy/HS/HS_1/HS_1_1",
            }
        ],
    )

    cfg = _config(raw, processed)
    cfg["validation"]["voisard"] = {"max_length_diff_rows": 0}

    with pytest.raises(RawDataValidationError):
        RawDataValidator(cfg).run()

    report = pd.read_csv(processed / "quality_report.csv")
    row = report[report["check"] == "sensor_length_consistency"].iloc[0]
    assert row["status"] == "fail"


def test_validate_raw_fails_invalid_daphnet_label(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)
    (daphnet_dir / "S01R01.txt").write_text(
        "15 70 39 -970 0 0 0 0 0 0 9\n",
        encoding="utf-8",
    )

    _write_inventory(
        processed,
        [
            {
                "dataset": "daphnet",
                "subject": "S01",
                "cohort": "PD",
                "trial": "S01R01",
                "source_path": "daphnet/S01R01.txt",
            }
        ],
    )

    with pytest.raises(RawDataValidationError):
        RawDataValidator(_config(raw, processed)).run()


def test_validate_raw_allows_uint16_packet_counter_wrap(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    trial_dir = raw / "voisard" / "healthy" / "HS" / "HS_1" / "HS_1_1"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "HS_1_1_meta.json").write_text(
        '{"subject":"HS_1","pathologyKey":"HS"}', encoding="utf-8"
    )
    lines = ["PacketCounter\tAcc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z"]
    counters = [65534, 65535, 0, 1, 2]
    for counter in counters:
        lines.append(f"{counter}\t1.0\t2.0\t3.0\t4.0\t5.0\t6.0\t7.0\t8.0\t9.0")
    for code in ["HE", "LB", "LF", "RF"]:
        (trial_dir / f"HS_1_1_raw_data_{code}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    _write_inventory(
        processed,
        [
            {
                "dataset": "voisard",
                "subject": "HS_1",
                "cohort": "Healthy",
                "trial": "HS_1_1",
                "source_path": "voisard/healthy/HS/HS_1/HS_1_1",
            }
        ],
    )

    report = RawDataValidator(_config(raw, processed)).run()
    assert (report["status"] == "fail").sum() == 0


def test_validate_raw_can_continue_when_fail_on_error_disabled(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)
    (daphnet_dir / "S01R01.txt").write_text("bad row\n", encoding="utf-8")

    _write_inventory(
        processed,
        [
            {
                "dataset": "daphnet",
                "subject": "S01",
                "cohort": "PD",
                "trial": "S01R01",
                "source_path": "daphnet/S01R01.txt",
            }
        ],
    )

    report = RawDataValidator(_config(raw, processed, fail_on_error=False)).run()
    assert (report["status"] == "fail").any()
