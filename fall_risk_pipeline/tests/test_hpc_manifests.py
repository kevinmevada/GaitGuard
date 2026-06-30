"""Tests for HPC manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.hpc.manifests import write_ingest_manifests


def _config(tmp_path: Path) -> dict:
    pipeline = tmp_path / "fall_risk_pipeline"
    configs = pipeline / "configs"
    configs.mkdir(parents=True)
    processed = pipeline / "data" / "processed"
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
        "_pipeline_meta": {"config_path": str(configs / "pipeline_config.yaml")},
        "paths": {"processed_data": str(processed)},
        "hpc": {
            "manifests_dir": "data/hpc/manifests",
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
    # manifests must live under trial processed root, not configs/
    assert "configs" not in str(paths[0])
