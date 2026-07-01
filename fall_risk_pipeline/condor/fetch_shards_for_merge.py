#!/usr/bin/env python3
"""Unpack HTCondor-transferred shard tarballs for merge jobs."""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


def _unpack_transferred_shards(stage: str) -> int:
    scratch = Path.cwd()
    n = 0
    for tar_path in sorted(scratch.glob("chunk_*.tar.gz")):
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(scratch)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)
    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0
    count = _unpack_transferred_shards(args.stage)
    print(f"unpacked {count} shard tarballs for {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
