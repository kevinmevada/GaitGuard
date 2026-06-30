"""Tests for HPC manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.hpc.manifests import write_ingest_manifests


def _config(tmp_path: Path) -> dict:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    inv = pd.DataFrame(
        {
            "dataset": ["voisard"] * 5,
            "trial": [f"T_{i}" for i in range(5)],
            "complete": [True] * 5,
        }
    )
    inv.to_csv(processed / "dataset_inventory.csv", index=False)
    return {
        "paths": {"processed_data": str(processed)},
        "hpc": {
            "manifests_dir": str(tmp_path / "manifests"),
            "trials_per_chunk": 2,
        },
    }


def test_write_ingest_manifests_chunks(tmp_path: Path):
    config = _config(tmp_path)
    paths = write_ingest_manifests(config)
    assert len(paths) == 3  # 5 trials / 2 per chunk
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert "trial_ids" in payload
    assert len(payload["trial_ids"]) == 2
