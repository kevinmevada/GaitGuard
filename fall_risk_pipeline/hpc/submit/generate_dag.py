#!/usr/bin/env python3
"""
Generate HTCondor DAG for sharded GaitGuard pipeline.

Usage (on ap40, after discover):
  python hpc.py init
  python hpc.py manifests ingest
  python hpc.py manifests preprocess
  python hpc.py manifests features
  python hpc.py manifests anomaly
  python hpc/submit/generate_dag.py --config configs/pipeline_config.yaml --full
  condor_submit_dag condor/dags/gaitguard_sharded.dag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DAG_OUT = REPO / "condor" / "dags" / "gaitguard_sharded.dag"


def _manifests(config_path: Path, kind: str) -> list[Path]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hpc = cfg.get("hpc") or {}
    man_dir = Path(hpc.get("manifests_dir", "condor/manifests"))
    if not man_dir.is_absolute():
        man_dir = REPO / man_dir
    pattern = {
        "ingest": "ingest_chunk_*.json",
        "preprocess": "preprocess_chunk_*.json",
        "features": "features_chunk_*.json",
        "anomaly": "anomaly_fold_*.json",
    }[kind]
    return sorted(man_dir.glob(pattern))


def _job_block(name: str, sub: str, vars: dict[str, str]) -> list[str]:
    lines = [f"JOB {name} {sub}"]
    for k, v in vars.items():
        lines.append(f'VARS {name} {k}="{v}"')
    return lines


def _chunk_id_from_manifest(path: Path, kind: str) -> str:
    # ingest_chunk_0000.json -> chunk_0000
    stem = path.stem
    prefix = f"{kind}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _ingest_osdf_inputs(manifest_path: Path, cfg: dict) -> str:
    """Comma-separated osdf:// URLs for HTCondor transfer_input_files (per trial dir)."""
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    hpc = cfg.get("hpc") or {}
    staging = str(hpc.get("staging_root", "/ospool/ap40/data/kevin.mevada")).rstrip("/")
    base = f"osdf://{staging}/gaitguard/raw"
    urls: list[str] = []
    for rel in (man.get("trial_source_paths") or {}).values():
        rel = str(rel).strip("/")
        urls.append(f"{base}/{rel}?recursive")
    return ", ".join(urls)


def _osd_gg_base(cfg: dict) -> str:
    hpc = cfg.get("hpc") or {}
    staging = str(hpc.get("staging_root", "/ospool/ap40/data/kevin.mevada")).rstrip("/")
    return f"osdf://{staging}/gaitguard"


_SENSORS = ("head", "lower_back", "left_foot", "right_foot")


def _shard_osdf_inputs(manifest_path: Path, cfg: dict, stage: str) -> str:
    """osdf:// URLs for preprocess/features shard inputs (metadata + per-trial parquets)."""
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    base = _osd_gg_base(cfg)
    urls = [f"{base}/processed/trial_metadata.csv"]
    subdir = "signals" if stage == "preprocess" else "signals_clean"
    for trial_id in man.get("trial_ids", []):
        tid = str(trial_id)
        for sensor in _SENSORS:
            urls.append(f"{base}/processed/{subdir}/{tid}_{sensor}.parquet")
    return ", ".join(urls)


def _merge_osdf_shard_inputs(config_path: Path, cfg: dict, stage: str) -> str:
    """osdf:// URLs for all shard tarballs needed by a merge job."""
    kind = {"ingest": "ingest", "preprocess": "preprocess", "features": "features"}[stage]
    base = _osd_gg_base(cfg)
    urls: list[str] = []
    for m in _manifests(config_path, kind):
        cid = _chunk_id_from_manifest(m, kind)
        urls.append(f"{base}/hpc/shards/{stage}/{cid}.tar.gz")
    return ", ".join(urls)


def write_dag(config_path: Path, output: Path, *, include_downstream: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ingest_m = _manifests(config_path, "ingest")
    pre_m = _manifests(config_path, "preprocess")
    feat_m = _manifests(config_path, "features")
    anom_m = _manifests(config_path, "anomaly")

    if not ingest_m:
        raise SystemExit("No ingest manifests — run: python hpc.py manifests ingest")

    lines: list[str] = ["# Auto-generated sharded GaitGuard DAG"]

    ingest_jobs: list[str] = []
    for i, m in enumerate(ingest_m):
        name = f"ing_{i}"
        rel = m.relative_to(REPO).as_posix()
        cid = _chunk_id_from_manifest(m, "ingest")
        lines.extend(
            _job_block(
                name,
                "condor/hpc_shard_ingest.sub",
                {
                    "shard_manifest": rel,
                    "chunk_id": cid,
                    "osdf_inputs": _ingest_osdf_inputs(m, cfg),
                },
            )
        )
        ingest_jobs.append(name)

    lines.extend(
        _job_block(
            "merge_ingest",
            "condor/hpc_merge.sub",
            {
                "command": "merge",
                "stage": "ingest",
                "osdf_shard_inputs": _merge_osdf_shard_inputs(config_path, cfg, "ingest"),
            },
        )
    )
    lines.append(f"PARENT {' '.join(ingest_jobs)} CHILD merge_ingest")
    lines.append("PARENT merge_ingest CHILD extract_ingest")

    lines.extend(
        _job_block(
            "extract_ingest",
            "condor/extract_merge.sub",
            {"stage": "ingest"},
        )
    )

    pre_jobs: list[str] = []
    for i, m in enumerate(pre_m):
        name = f"pre_{i}"
        rel = m.relative_to(REPO).as_posix()
        cid = _chunk_id_from_manifest(m, "preprocess")
        lines.extend(
            _job_block(
                name,
                "condor/hpc_shard_cpu.sub",
                {
                    "stage": "preprocess",
                    "shard_manifest": rel,
                    "chunk_id": cid,
                    "osdf_inputs": _shard_osdf_inputs(m, cfg, "preprocess"),
                },
            )
        )
        pre_jobs.append(name)
    lines.extend(
        _job_block(
            "merge_preprocess",
            "condor/hpc_merge.sub",
            {
                "command": "merge",
                "stage": "preprocess",
                "osdf_shard_inputs": _merge_osdf_shard_inputs(config_path, cfg, "preprocess"),
            },
        )
    )
    if pre_jobs:
        lines.append(f"PARENT extract_ingest CHILD {' '.join(pre_jobs)}")
        lines.append(f"PARENT {' '.join(pre_jobs)} CHILD merge_preprocess")
        lines.append("PARENT merge_preprocess CHILD extract_preprocess")
    else:
        lines.append("PARENT extract_ingest CHILD merge_preprocess")
        lines.append("PARENT merge_preprocess CHILD extract_preprocess")

    lines.extend(
        _job_block(
            "extract_preprocess",
            "condor/extract_merge.sub",
            {"stage": "preprocess"},
        )
    )

    feat_jobs: list[str] = []
    for i, m in enumerate(feat_m):
        name = f"feat_{i}"
        rel = m.relative_to(REPO).as_posix()
        cid = _chunk_id_from_manifest(m, "features")
        lines.extend(
            _job_block(
                name,
                "condor/hpc_shard_cpu.sub",
                {
                    "stage": "features",
                    "shard_manifest": rel,
                    "chunk_id": cid,
                    "osdf_inputs": _shard_osdf_inputs(m, cfg, "features"),
                },
            )
        )
        feat_jobs.append(name)
    lines.extend(
        _job_block(
            "merge_features",
            "condor/hpc_merge.sub",
            {
                "command": "merge",
                "stage": "features",
                "osdf_shard_inputs": _merge_osdf_shard_inputs(config_path, cfg, "features"),
            },
        )
    )
    lines.extend(
        _job_block(
            "extract_features",
            "condor/extract_merge.sub",
            {"stage": "features"},
        )
    )
    if feat_jobs:
        lines.append(f"PARENT extract_preprocess CHILD {' '.join(feat_jobs)}")
        lines.append(f"PARENT {' '.join(feat_jobs)} CHILD merge_features")
        lines.append("PARENT merge_features CHILD extract_features")
    else:
        lines.append("PARENT extract_preprocess CHILD merge_features")
        lines.append("PARENT merge_features CHILD extract_features")

    anom_jobs: list[str] = []
    for i, m in enumerate(anom_m):
        name = f"anom_{i}"
        rel = m.relative_to(REPO).as_posix()
        lines.extend(_job_block(name, "condor/hpc_shard_gpu.sub", {"shard_manifest": rel}))
        anom_jobs.append(name)
    lines.extend(_job_block("reduce_anomaly", "condor/hpc_merge_reduce.sub", {"command": "reduce", "stage": "anomaly"}))
    if anom_jobs:
        lines.append(f"PARENT extract_features CHILD {' '.join(anom_jobs)}")
        lines.append(f"PARENT {' '.join(anom_jobs)} CHILD reduce_anomaly")
    else:
        lines.append("PARENT extract_features CHILD reduce_anomaly")

    last = "reduce_anomaly"
    if include_downstream:
        for st in [
            "eda",
            "phase3_features",
            "select_features",
            "train",
            "evaluate",
            "ablation",
            "classical_baselines",
            "competitor_metrics",
            "statistical_benchmark",
            "compute_overhead",
            "novelty_table",
            "per_cohort_loso",
            "fall_risk_spearman",
            "cross_cohort",
            "predict",
        ]:
            j = f"st_{st}"
            lines.extend(_job_block(j, "condor/cpu_stage.sub", {"stage": st}))
            lines.append(f"PARENT {last} CHILD {j}")
            last = j
        for st in ["train_deep", "sensor_ablation", "dl_baselines", "severity_regression"]:
            j = f"st_{st}"
            lines.extend(_job_block(j, "condor/gpu_stage.sub", {"stage": st}))
            lines.append(f"PARENT reduce_anomaly CHILD {j}")
        lines.extend(_job_block("st_report", "condor/cpu_stage.sub", {"stage": "report"}))
        lines.append(f"PARENT {last} CHILD st_report")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {output} — {len(ingest_jobs)} ingest, {len(pre_jobs)} preprocess, "
        f"{len(feat_jobs)} features, {len(anom_jobs)} anomaly folds"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--output", default=str(DAG_OUT))
    parser.add_argument("--full", action="store_true", help="Append single-job downstream stages")
    args = parser.parse_args()
    write_dag(Path(args.config), Path(args.output), include_downstream=args.full)


if __name__ == "__main__":
    main()
