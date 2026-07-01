#!/usr/bin/env python3
"""Stage per-shard inputs from OSDF onto the worker scratch directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger


def _osdf_copy(src: str, dst: Path, *, recursive: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    flags = ["-r"] if recursive else []
    for url in (src, src if src.startswith("osdf://") else f"osdf://{src.lstrip('/')}"):
        try:
            subprocess.run(["stashcp", *flags, url, str(dst)], check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.warning("stashcp failed {} -> {}: {}", url, dst, exc)
    return False


def _stage_ingest(manifest: dict, osd_gg: str, raw_root: Path) -> None:
    paths = manifest.get("trial_source_paths") or {}
    ok = 0
    scratch = Path.cwd()
    for trial_id, rel in paths.items():
        rel = str(rel).strip("/")
        trial_dir = raw_root / rel
        if trial_dir.is_dir():
            ok += 1
            continue
        # HTCondor osdf:// transfer may land at cwd/<rel>, cwd/<basename>, etc.
        linked = False
        for cand in (scratch / rel, scratch / Path(rel).name):
            if cand.is_dir():
                trial_dir.parent.mkdir(parents=True, exist_ok=True)
                if not trial_dir.exists():
                    cand.rename(trial_dir)
                linked = True
                ok += 1
                break
        if linked:
            continue
        src = f"{osd_gg}/raw/{rel}?recursive"
        if _osdf_copy(src, trial_dir, recursive=True):
            ok += 1
        else:
            logger.warning("ingest staging miss: trial {} ({})", trial_id, rel)
    logger.info("ingest staging: {}/{} trial dirs ready", ok, len(paths))


_SENSORS = ("head", "lower_back", "left_foot", "right_foot")


def _link_transferred_file(scratch: Path, dst: Path, *names: str) -> bool:
    if dst.is_file():
        return True
    for name in names:
        cand = scratch / name
        if cand.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            cand.rename(dst)
            return True
    return False


def _stage_preprocess_or_features(
    manifest: dict, osd_gg: str, proc_root: Path, *, signals: bool
) -> None:
    scratch = Path.cwd()
    meta_local = proc_root / "trial_metadata.csv"
    if not meta_local.is_file():
        if not _link_transferred_file(scratch, meta_local, "trial_metadata.csv"):
            _osdf_copy(f"{osd_gg}/processed/trial_metadata.csv", meta_local)
    if not signals:
        return
    signals_dir = proc_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in manifest.get("trial_ids", []):
        tid = str(trial_id)
        for sensor in _SENSORS:
            name = f"{tid}_{sensor}.parquet"
            dst = signals_dir / name
            if dst.is_file():
                continue
            if not _link_transferred_file(scratch, dst, name):
                src = f"{osd_gg}/processed/signals/{name}"
                _osdf_copy(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    osd_gg = os.environ.get("OSDF_GAITGUARD", "osdf:///ospool/ap40/data/kevin.mevada/gaitguard")
    raw_root = Path("data/raw")
    proc_root = Path("data/processed")
    raw_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "ingest":
        _stage_ingest(manifest, osd_gg, raw_root)
    elif args.stage == "preprocess":
        _stage_preprocess_or_features(manifest, osd_gg, proc_root, signals=True)
    else:
        _stage_preprocess_or_features(manifest, osd_gg, proc_root, signals=False)
        signals_clean = proc_root / "signals_clean"
        signals_clean.mkdir(parents=True, exist_ok=True)
        scratch = Path.cwd()
        for trial_id in manifest.get("trial_ids", []):
            tid = str(trial_id)
            for sensor in _SENSORS:
                name = f"{tid}_{sensor}.parquet"
                dst = signals_clean / name
                if dst.is_file():
                    continue
                if not _link_transferred_file(scratch, dst, name):
                    src = f"{osd_gg}/processed/signals_clean/{name}"
                    _osdf_copy(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
