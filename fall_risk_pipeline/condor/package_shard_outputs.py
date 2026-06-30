#!/usr/bin/env python3
"""Tar shard output and upload to OSDF via stashcp (no HTCondor output remap)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _scratch_root() -> Path:
    return Path(os.environ.get("_CONDOR_SCRATCH_DIR", ".")).resolve()


def _push_osdf(local: Path, osd_dest: str) -> None:
    subprocess.run(["stashcp", str(local), osd_dest], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features", "anomaly"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--error", default="", help="Optional error message from shard run")
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    scratch = _scratch_root()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    osd_gg = os.environ.get("OSDF_GAITGUARD", "osdf:///ospool/ap40/data/kevin.mevada/gaitguard")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=scratch, delete=False) as tmp:
        tar_path = Path(tmp.name)

    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            if args.stage == "anomaly":
                pid = manifest.get("held_out_participant_id", "fold")
                safe = str(pid).replace("/", "_")
                src = scratch / "data/hpc/oof/anomaly" / f"fold_{safe}.parquet"
                if src.is_file():
                    tf.add(src, arcname=src.relative_to(scratch).as_posix())
            else:
                chunk_id = manifest.get("chunk_id")
                if not chunk_id:
                    raise SystemExit("manifest missing chunk_id")
                src = scratch / "data/hpc/shards" / args.stage / chunk_id
                if src.is_dir():
                    for f in sorted(src.rglob("*")):
                        if f.is_file():
                            tf.add(f, arcname=f.relative_to(scratch).as_posix())
                elif args.error:
                    err = scratch / "shard_error.json"
                    err.write_text(
                        json.dumps(
                            {
                                "stage": args.stage,
                                "chunk_id": chunk_id,
                                "error": args.error,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    tf.add(err, arcname="shard_error.json")
                else:
                    raise SystemExit(f"shard output missing: {src}")

        if args.stage == "anomaly":
            pid = manifest.get("held_out_participant_id", "fold")
            safe = str(pid).replace("/", "_")
            osd_dest = f"{osd_gg}/hpc/oof/anomaly/fold_{safe}.tar.gz"
        else:
            chunk_id = manifest["chunk_id"]
            osd_dest = f"{osd_gg}/hpc/shards/{args.stage}/{chunk_id}.tar.gz"

        _push_osdf(tar_path, osd_dest)
        print(f"uploaded {tar_path} -> {osd_dest}")
        if args.error:
            return 1
        return 0
    finally:
        tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
