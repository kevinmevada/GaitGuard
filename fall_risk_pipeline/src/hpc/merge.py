"""Merge shard outputs into canonical processed paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from loguru import logger

from src.hpc.paths import shard_root


def merge_ingest_shards(config: dict, *, include_daphnet: bool = True) -> Path:
    """
    Concatenate Voisard ingest shards → ``data/processed/signals/`` + ``trial_metadata.csv``.
    """
    from src.ingestion.data_loader import DataLoader

    root = shard_root(config) / "ingest"
    if not root.is_dir():
        raise FileNotFoundError(f"No ingest shards under {root}")

    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    meta_chunks: list[pd.DataFrame] = []
    gap_chunks: list[pd.DataFrame] = []

    for chunk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = chunk_dir / "trial_metadata_chunk.csv"
        if meta_path.is_file():
            meta_chunks.append(pd.read_csv(meta_path))
        chunk_signals = chunk_dir / "signals"
        if chunk_signals.is_dir():
            for src in chunk_signals.glob("*.parquet"):
                dst = signals_dir / src.name
                shutil.copy2(src, dst)
        ge_src = chunk_dir / "gait_events"
        if ge_src.is_dir():
            ge_dst = processed / "gait_events"
            ge_dst.mkdir(exist_ok=True)
            for src in ge_src.glob("*.csv"):
                shutil.copy2(src, ge_dst / src.name)
        gap_path = chunk_dir / "packet_gap_chunk.csv"
        if gap_path.is_file():
            gap_chunks.append(pd.read_csv(gap_path))

    if not meta_chunks:
        raise RuntimeError(f"No trial_metadata_chunk.csv found under {root}")

    meta_df = pd.concat(meta_chunks, ignore_index=True).drop_duplicates("trial_id")
    meta_path = processed / "trial_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info("Merged ingest metadata → {} ({} trials)", meta_path, len(meta_df))

    loader = DataLoader(config)
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if gap_chunks:
        gap_df = pd.concat(gap_chunks, ignore_index=True)
        gap_df.to_csv(metrics_dir / "packet_gap_report.csv", index=False)
    loader._write_packet_gap_report(len(meta_df))  # noqa: SLF001 — reuse summary writer

    if include_daphnet:
        daphnet_records = loader.run_daphnet_ingest()
        loader.append_daphnet_to_processed(daphnet_records)
        try:
            from src.dataset.label_balance import save_class_distribution_reports

            full_meta = pd.read_csv(meta_path)
            if "participant_id" in full_meta.columns:
                participants = full_meta.drop_duplicates("participant_id")
                save_class_distribution_reports(participants, metrics_dir, config=config)
            else:
                save_class_distribution_reports(full_meta, metrics_dir, config=config)
        except Exception as exc:
            logger.warning("Class distribution report skipped: {}", exc)

    return meta_path


def merge_preprocess_shards(config: dict) -> None:
    """Collect per-chunk ``signals_clean/`` + uturn reports into processed paths."""
    root = shard_root(config) / "preprocess"
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed = Path(config["paths"]["processed_data"])
    clean_dir = processed / "signals_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    n_clean = 0
    if root.is_dir():
        for chunk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            chunk_clean = chunk_dir / "signals_clean"
            if chunk_clean.is_dir():
                for src in chunk_clean.glob("*.parquet"):
                    shutil.copy2(src, clean_dir / src.name)
                    n_clean += 1
            uturn = chunk_dir / "uturn_exclusion_chunk.csv"
            if uturn.is_file():
                rows.append(pd.read_csv(uturn))
    logger.info("Merged {} cleaned signal files → {}", n_clean, clean_dir)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            metrics_dir / "uturn_exclusion_report.csv", index=False
        )
        logger.info("Merged uturn exclusion report ({} chunks)", len(rows))
    else:
        logger.info("No preprocess uturn shard reports to merge")


def merge_features_shards(config: dict) -> Path:
    """Concat trial feature shards and rebuild patient-level parquet."""
    from src.features.feature_extractor import FeatureExtractor

    root = shard_root(config) / "features"
    if not root.is_dir():
        raise FileNotFoundError(f"No feature shards under {root}")

    chunks: list[pd.DataFrame] = []
    for chunk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        part = chunk_dir / "trial_features_chunk.parquet"
        if part.is_file():
            chunks.append(pd.read_parquet(part))

    if not chunks:
        raise RuntimeError(f"No trial_features_chunk.parquet under {root}")

    trial_df = pd.concat(chunks, ignore_index=True).drop_duplicates("trial_id")
    feat_dir = Path(config["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    trial_path = feat_dir / "trial_features.parquet"
    trial_df.to_parquet(trial_path, index=False)
    logger.info("Merged trial features → {} shape={}", trial_path, trial_df.shape)

    extractor = FeatureExtractor(config)
    patient_df = extractor.aggregate_patient_features(trial_df)
    patient_path = feat_dir / "patient_features.parquet"
    patient_df.to_parquet(patient_path, index=False)
    logger.info("Patient features → {} shape={}", patient_path, patient_df.shape)
    return trial_path
