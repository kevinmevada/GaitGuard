"""Tests for Stage 0 dataset discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.dataset_discovery import DatasetDiscovery


def _config(raw: Path, processed: Path) -> dict:
    return {
        "paths": {
            "raw_data": str(raw),
            "processed_data": str(processed),
        },
        "discovery": {
            "sources": ["voisard", "daphnet"],
            "inventory_path": str(processed / "dataset_inventory.csv"),
        },
    }


def _write_voisard_trial(
    trial_dir: Path,
    *,
    trial_id: str = "HS_1_1",
    subject: str = "HS_1",
    pathology_key: str = "HS",
    sensors: list[str] | None = None,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "subject": subject,
        "pathologyKey": pathology_key,
        "trial": 1,
    }
    (trial_dir / f"{trial_id}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    for code in sensors or ["HE", "LB", "LF", "RF"]:
        (trial_dir / f"{trial_id}_raw_data_{code}.txt").write_text(
            "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0\n",
            encoding="utf-8",
        )


def test_discover_voisard_and_daphnet(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    voisard_trial = raw / "voisard" / "healthy" / "HS" / "HS_1" / "HS_1_1"
    _write_voisard_trial(voisard_trial)

    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)
    (daphnet_dir / "S01R01.txt").write_text("15 70 39 -970 0 0 0 0 0 0 0\n", encoding="utf-8")

    df = DatasetDiscovery(_config(raw, processed)).run()

    assert len(df) == 2
    voisard_row = df[df["dataset"] == "voisard"].iloc[0]
    assert voisard_row["subject"] == "HS_1"
    assert voisard_row["cohort"] == "Healthy"
    assert voisard_row["trial"] == "HS_1_1"
    assert voisard_row["sensors"] == "HE LB LF RF"
    assert bool(voisard_row["complete"]) is True

    daphnet_row = df[df["dataset"] == "daphnet"].iloc[0]
    assert daphnet_row["subject"] == "S01"
    assert daphnet_row["cohort"] == "PD"
    assert daphnet_row["trial"] == "S01R01"
    assert daphnet_row["sensors"] == "ANK TH TR"
    assert bool(daphnet_row["complete"]) is True

    out = processed / "dataset_inventory.csv"
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == list(df.columns)


def test_discover_flags_missing_voisard_sensors(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    trial_dir = raw / "voisard" / "neuro" / "PD" / "PD_1" / "PD_1_1"
    _write_voisard_trial(trial_dir, trial_id="PD_1_1", subject="PD_1", pathology_key="PD", sensors=["HE", "LB"])

    df = DatasetDiscovery(_config(raw, processed)).run()
    row = df.iloc[0]

    assert row["cohort"] == "PD"
    assert row["sensors"] == "HE LB"
    assert row["missing_sensors"] == "LF RF"
    assert bool(row["complete"]) is False


def test_discover_skips_non_daphnet_txt_files(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)
    (daphnet_dir / "README.txt").write_text("notes", encoding="utf-8")
    (daphnet_dir / "S02R01.txt").write_text("1 2 3 4 5 6 7 8 9 10 0\n", encoding="utf-8")

    df = DatasetDiscovery(_config(raw, processed)).run()
    assert len(df) == 1
    assert df.iloc[0]["trial"] == "S02R01"


@pytest.mark.skipif(
    not Path("data/raw/voisard").exists(),
    reason="local Voisard data not present",
)
def test_discover_on_local_voisard():
    raw = Path("data/raw")
    processed = Path("data/processed")
    df = DatasetDiscovery(
        {
            "paths": {"raw_data": str(raw), "processed_data": str(processed)},
            "discovery": {
                "sources": ["voisard"],
                "inventory_path": str(processed / "dataset_inventory_test.csv"),
            },
        }
    ).run()
    assert len(df) >= 1000
    assert df["complete"].mean() > 0.9
