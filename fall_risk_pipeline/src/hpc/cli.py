"""CLI for sharded HPC pipeline execution on OSPool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.hpc.anomaly import reduce_anomaly_folds, run_anomaly_fold
from src.hpc.manifests import (
    load_manifest,
    write_anomaly_fold_manifests,
    write_features_manifests,
    write_ingest_manifests,
    write_preprocess_manifests,
)
from src.hpc.merge import merge_features_shards, merge_ingest_shards, merge_preprocess_shards
from src.hpc.paths import (
    features_chunk_dir,
    ingest_chunk_dir,
    preprocess_chunk_dir,
    shard_root,
)
from src.utils.reproducibility import get_pipeline_seed, set_global_seed


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_pipeline_meta"] = {"config_path": str(Path(path).resolve())}
    return config


def cmd_manifests(args: argparse.Namespace, config: dict) -> None:
    kind = args.kind
    if kind == "ingest":
        paths = write_ingest_manifests(config)
    elif kind == "preprocess":
        paths = write_preprocess_manifests(config)
    elif kind == "features":
        paths = write_features_manifests(config)
    elif kind == "anomaly":
        paths = write_anomaly_fold_manifests(config)
    else:
        raise ValueError(f"Unknown manifest kind: {kind}")
    for p in paths[:5]:
        print(p)
    if len(paths) > 5:
        print(f"... and {len(paths) - 5} more")


def cmd_shard(args: argparse.Namespace, config: dict) -> None:
    manifest = load_manifest(args.manifest)
    stage = args.stage

    if stage == "ingest":
        from src.ingestion.data_loader import DataLoader

        chunk_id = manifest["chunk_id"]
        trial_ids = manifest["trial_ids"]
        out = ingest_chunk_dir(config, chunk_id)
        DataLoader(config).run_voisard_shard(trial_ids, out)

    elif stage == "preprocess":
        from src.preprocessing.signal_processor import SignalProcessor

        chunk_id = manifest["chunk_id"]
        trial_ids = manifest["trial_ids"]
        out = preprocess_chunk_dir(config, chunk_id)
        SignalProcessor(config).run_shard(trial_ids, out)

    elif stage == "features":
        from src.features.feature_extractor import FeatureExtractor

        chunk_id = manifest["chunk_id"]
        trial_ids = manifest["trial_ids"]
        out = features_chunk_dir(config, chunk_id)
        FeatureExtractor(config).run_shard(trial_ids, out)

    elif stage == "anomaly":
        pid = manifest["held_out_participant_id"]
        run_anomaly_fold(config, pid)

    else:
        raise ValueError(f"Unknown shard stage: {stage}")


def cmd_merge(args: argparse.Namespace, config: dict) -> None:
    stage = args.stage
    if stage == "ingest":
        merge_ingest_shards(config, include_daphnet=not args.skip_daphnet)
    elif stage == "preprocess":
        merge_preprocess_shards(config)
    elif stage == "features":
        merge_features_shards(config)
    else:
        raise ValueError(f"Unknown merge stage: {stage}")


def cmd_reduce(args: argparse.Namespace, config: dict) -> None:
    if args.stage == "anomaly":
        reduce_anomaly_folds(config)
    else:
        raise ValueError(f"Unknown reduce stage: {args.stage}")


def cmd_init_dirs(args: argparse.Namespace, config: dict) -> None:
    root = shard_root(config)
    for sub in ("ingest", "preprocess", "features"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    from src.hpc.paths import oof_root

    (oof_root(config) / "anomaly").mkdir(parents=True, exist_ok=True)
    logger.info("HPC shard directories ready under {}", root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GaitGuard HPC sharded execution (OSPool)")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create shard/oof directories")
    p_init.set_defaults(func=cmd_init_dirs)

    p_man = sub.add_parser("manifests", help="Write job manifest JSON files")
    p_man.add_argument("kind", choices=["ingest", "preprocess", "features", "anomaly"])
    p_man.set_defaults(func=cmd_manifests)

    p_shard = sub.add_parser("shard", help="Run one shard job")
    p_shard.add_argument("stage", choices=["ingest", "preprocess", "features", "anomaly"])
    p_shard.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p_shard.set_defaults(func=cmd_shard)

    p_merge = sub.add_parser("merge", help="Merge shard outputs")
    p_merge.add_argument("stage", choices=["ingest", "preprocess", "features"])
    p_merge.add_argument("--skip-daphnet", action="store_true")
    p_merge.set_defaults(func=cmd_merge)

    p_reduce = sub.add_parser("reduce", help="Reduce parallel fold outputs to metrics")
    p_reduce.add_argument("stage", choices=["anomaly"])
    p_reduce.set_defaults(func=cmd_reduce)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    set_global_seed(get_pipeline_seed(config))
    args.func(args, config)


if __name__ == "__main__":
    main()
