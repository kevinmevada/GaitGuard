#!/usr/bin/env python3
"""Tar shard output for HTCondor transfer_output_remaps → OSDF."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features", "anomaly"])
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.stage == "anomaly":
        pid = manifest.get("held_out_participant_id", "fold")
        safe = str(pid).replace("/", "_")
        src = Path("data/hpc/oof/anomaly") / f"fold_{safe}.parquet"
        out = Path("shard_out.tar.gz")
        with tarfile.open(out, "w:gz") as tf:
            if src.is_file():
                tf.add(src, arcname=str(src))
        return 0

    chunk_id = manifest.get("chunk_id")
    if not chunk_id:
        raise SystemExit("manifest missing chunk_id")
    src = Path("data/hpc/shards") / args.stage / chunk_id
    out = Path("shard_out.tar.gz")
    if not src.is_dir():
        raise SystemExit(f"shard output missing: {src}")
    subprocess.run(["tar", "-czf", str(out), "-C", str(src.parent), chunk_id], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
