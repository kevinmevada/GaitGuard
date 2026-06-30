"""Build trial / participant manifests for HTCondor array jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.hpc.paths import chunk_size, manifests_dir


def _load_inventory(config: dict) -> pd.DataFrame:
    inv_path = Path(config["paths"]["processed_data"]) / "dataset_inventory.csv"
    if not inv_path.is_file():
        raise FileNotFoundError(
            f"Missing {inv_path}. Run stage 'discover' before generating manifests."
        )
    df = pd.read_csv(inv_path)
    return df[df["dataset"].astype(str).str.lower() == "voisard"].reset_index(drop=True)


def _chunk_list(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def write_ingest_manifests(config: dict) -> list[Path]:
    """Write ``ingest_chunk_NNN.json`` files; return paths."""
    inv = _load_inventory(config)
    trial_ids = sorted(inv["trial"].astype(str).unique().tolist())
    out_dir = manifests_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = chunk_size(config)
    paths: list[Path] = []
    for idx, chunk in enumerate(_chunk_list(trial_ids, size)):
        chunk_id = f"chunk_{idx:04d}"
        payload = {"chunk_id": chunk_id, "trial_ids": chunk}
        path = out_dir / f"ingest_{chunk_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    logger.info("Wrote {} ingest manifests ({} trials, chunk size {})", len(paths), len(trial_ids), size)
    return paths


def _voisard_trial_ids(config: dict) -> list[str]:
    """Trial IDs for shard manifests (inventory before ingest merge, metadata after)."""
    meta_path = Path(config["paths"]["processed_data"]) / "trial_metadata.csv"
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        return sorted(
            meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")]["trial_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    logger.warning(
        "{} not found — using dataset_inventory trial IDs (same as ingest manifests)",
        meta_path,
    )
    inv = _load_inventory(config)
    return sorted(inv["trial"].astype(str).unique().tolist())


def write_preprocess_manifests(config: dict) -> list[Path]:
    trial_ids = _voisard_trial_ids(config)
    out_dir = manifests_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = chunk_size(config)
    paths: list[Path] = []
    for idx, chunk in enumerate(_chunk_list(trial_ids, size)):
        chunk_id = f"chunk_{idx:04d}"
        payload = {"chunk_id": chunk_id, "trial_ids": chunk}
        path = out_dir / f"preprocess_{chunk_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    logger.info("Wrote {} preprocess manifests", len(paths))
    return paths


def write_features_manifests(config: dict) -> list[Path]:
    trial_ids = _voisard_trial_ids(config)
    out_dir = manifests_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = chunk_size(config)
    paths: list[Path] = []
    for idx, chunk in enumerate(_chunk_list(trial_ids, size)):
        chunk_id = f"chunk_{idx:04d}"
        payload = {"chunk_id": chunk_id, "trial_ids": chunk}
        path = out_dir / f"features_{chunk_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    logger.info("Wrote {} features manifests", len(paths))
    return paths


def write_anomaly_fold_manifests(config: dict) -> list[Path]:
    meta_path = Path(config["paths"]["processed_data"]) / "trial_metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}.")
    meta = pd.read_csv(meta_path)
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")]
    participant_ids = sorted(meta["participant_id"].astype(str).unique().tolist())
    out_dir = manifests_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pid in participant_ids:
        payload = {"held_out_participant_id": pid}
        safe = pid.replace("/", "_")
        path = out_dir / f"anomaly_fold_{safe}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    logger.info("Wrote {} anomaly fold manifests", len(paths))
    return paths


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
