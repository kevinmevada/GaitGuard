"""Resolve HPC shard / manifest / OOF paths from pipeline config."""

from __future__ import annotations

from pathlib import Path


def hpc_cfg(config: dict) -> dict:
    return config.get("hpc") or {}


def pipeline_root(config: dict) -> Path:
    meta = config.get("_pipeline_meta") or {}
    cfg_path = meta.get("config_path")
    if cfg_path:
        return Path(cfg_path).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_hpc_path(config: dict, key: str) -> Path:
    rel = hpc_cfg(config).get(key, f"data/hpc/{key}")
    p = Path(rel)
    if p.is_absolute():
        return p
    return pipeline_root(config) / p


def shard_root(config: dict) -> Path:
    return resolve_hpc_path(config, "shard_root")


def oof_root(config: dict) -> Path:
    return resolve_hpc_path(config, "oof_root")


def manifests_dir(config: dict) -> Path:
    return resolve_hpc_path(config, "manifests_dir")


def chunk_dir(config: dict, stage: str, chunk_id: str) -> Path:
    return shard_root(config) / stage / chunk_id


def ingest_chunk_dir(config: dict, chunk_id: str) -> Path:
    return chunk_dir(config, "ingest", chunk_id)


def preprocess_chunk_dir(config: dict, chunk_id: str) -> Path:
    return chunk_dir(config, "preprocess", chunk_id)


def features_chunk_dir(config: dict, chunk_id: str) -> Path:
    return chunk_dir(config, "features", chunk_id)


def anomaly_fold_path(config: dict, participant_id: str) -> Path:
    safe = str(participant_id).replace("/", "_")
    return oof_root(config) / "anomaly" / f"fold_{safe}.parquet"


def chunk_size(config: dict) -> int:
    return int(hpc_cfg(config).get("trials_per_chunk", 20))
