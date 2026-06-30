#!/usr/bin/env python3
"""Fetch shard tarballs from OSDF and unpack for merge jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def _osdf_list_and_fetch(stage: str, osd_gg: str, shard_root: Path) -> int:
    prefix = f"{osd_gg}/hpc/shards/{stage}/"
    listing = Path("shard_listing.txt")
    try:
        subprocess.run(["stashcp", f"{prefix}", str(listing)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0

    shard_root.mkdir(parents=True, exist_ok=True)
    stage_dir = shard_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Fallback: try known chunk pattern from local manifests if listing fails.
    man_dir = Path("data/hpc/manifests")
    if man_dir.is_dir():
        for manifest in sorted(man_dir.glob(f"{stage}_chunk_*.json")):
            chunk_id = manifest.stem.split("_", 1)[1]
            tar_name = f"{chunk_id}.tar.gz"
            local_tar = Path(tar_name)
            if not local_tar.is_file():
                try:
                    subprocess.run(
                        ["stashcp", f"{prefix}{tar_name}", str(local_tar)],
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    continue
            out_dir = stage_dir / chunk_id
            if out_dir.is_dir():
                continue
            if local_tar.is_file():
                with tarfile.open(local_tar, "r:gz") as tf:
                    tf.extractall(stage_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)
    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0
    osd_gg = os.environ.get("OSDF_GAITGUARD", "osdf:///ospool/ap40/data/kevin.mevada/gaitguard")
    return _osdf_list_and_fetch(args.stage, osd_gg, Path("data/hpc/shards"))


if __name__ == "__main__":
    sys.exit(main())
