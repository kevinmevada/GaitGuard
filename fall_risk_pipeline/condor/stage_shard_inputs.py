#!/usr/bin/env python3
"""Stage per-shard inputs from OSDF onto the worker scratch directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _osdf_copy(src: str, dst: Path, *, recursive: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    flags = ["-r"] if recursive else []
    cmd = ["stashcp", *flags, src, str(dst)]
    try:
        subprocess.run(cmd, check=True)
        return
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass
    # HTCondor OSDF plugin URL — stashcp preferred on OSPool when available.
    url = src if src.startswith("osdf://") else f"osdf://{src.lstrip('/')}"
    cmd = ["stashcp", *flags, url, str(dst)]
    subprocess.run(cmd, check=True)


def _stage_ingest(manifest: dict, osd_gg: str, raw_root: Path) -> None:
    paths = manifest.get("trial_source_paths") or {}
    for _trial_id, rel in paths.items():
        rel = str(rel).strip("/")
        trial_dir = raw_root / rel
        if trial_dir.is_dir():
            continue
        src = f"{osd_gg}/raw/{rel}?recursive"
        _osdf_copy(src, trial_dir, recursive=True)


def _stage_preprocess_or_features(
    manifest: dict, osd_gg: str, proc_root: Path, *, signals: bool
) -> None:
    meta_local = proc_root / "trial_metadata.csv"
    if not meta_local.is_file():
        _osdf_copy(f"{osd_gg}/processed/trial_metadata.csv", meta_local)
    if not signals:
        return
    for trial_id in manifest.get("trial_ids", []):
        tid = str(trial_id)
        for pattern in (f"{tid}_*.parquet",):
            # Signals are per-sensor files; fetch whole trial prefix via listing is expensive.
            # Copy from merged signals dir on OSDF if present.
            pass
    signals_dir = proc_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in manifest.get("trial_ids", []):
        tid = str(trial_id)
        for sensor in ("head", "lower_back", "left_foot", "right_foot"):
            name = f"{tid}_{sensor}.parquet"
            dst = signals_dir / name
            if dst.is_file():
                continue
            src = f"{osd_gg}/processed/signals/{name}"
            try:
                _osdf_copy(src, dst)
            except subprocess.CalledProcessError:
                continue


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
        for trial_id in manifest.get("trial_ids", []):
            tid = str(trial_id)
            for sensor in ("head", "lower_back", "left_foot", "right_foot"):
                name = f"{tid}_{sensor}.parquet"
                dst = signals_clean / name
                if dst.is_file():
                    continue
                src = f"{osd_gg}/processed/signals_clean/{name}"
                try:
                    _osdf_copy(src, dst)
                except subprocess.CalledProcessError:
                    continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
