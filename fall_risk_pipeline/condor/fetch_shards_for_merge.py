#!/usr/bin/env python3
"""Unpack HTCondor-transferred shard tarballs for merge jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

try:
    from local_paths import gaitguard_root, local_staging_enabled
except ImportError:  # OSPool worker without local_paths.py.
    def local_staging_enabled() -> bool:
        return False

    def gaitguard_root():  # pragma: no cover
        return Path(".")


def _unpack_transferred_shards(stage: str) -> int:
    scratch = Path.cwd()
    n = 0
    for tar_path in sorted(scratch.glob("chunk_*.tar.gz")):
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(scratch)
        n += 1
    return n


def _fetch_local_shards(stage: str) -> int:
    src_dir = gaitguard_root() / "hpc" / "shards" / stage
    if not src_dir.is_dir():
        return 0
    scratch = Path.cwd()
    n = 0
    for tar_path in sorted(src_dir.glob("chunk_*.tar.gz")):
        dst = scratch / tar_path.name
        if not dst.is_file():
            shutil.copy2(tar_path, dst)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)
    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0
    count = 0
    if local_staging_enabled():
        count = _fetch_local_shards(args.stage)
    count += _unpack_transferred_shards(args.stage)
    print(f"unpacked {count} shard tarballs for {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
