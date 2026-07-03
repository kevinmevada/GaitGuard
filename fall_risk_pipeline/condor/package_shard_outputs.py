#!/usr/bin/env python3
"""Tar shard output; upload via HTCondor remap (or stashcp for anomaly)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

try:
    from local_paths import gaitguard_root, local_staging_enabled
except ImportError:  # OSPool worker without local_paths.py.
    def local_staging_enabled() -> bool:
        return False

    def gaitguard_root():  # pragma: no cover
        return Path(".")


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

    # NEVER package/upload anything on failure. Uploading an error/partial
    # tarball poisons OSDF: the retry hits "remote object already exists" and
    # --skip-existing-ingest then counts the chunk as done. Instead leave
    # shard_out.tar.gz missing -> HTCondor holds on output transfer ->
    # periodic_release in the .sub re-runs the job on another site.
    if args.error:
        print(f"shard failed, not packaging output: {args.error}", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=scratch, delete=False) as tmp:
        tar_path = Path(tmp.name)

    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            if args.stage == "anomaly":
                pid = manifest.get("held_out_participant_id", "fold")
                safe = str(pid).replace("/", "_")
                src = scratch / "data/hpc/oof/anomaly" / f"fold_{safe}.parquet"
                if not src.is_file():
                    raise SystemExit(f"anomaly fold output missing: {src}")
                tf.add(src, arcname=src.relative_to(scratch).as_posix())
            else:
                chunk_id = manifest.get("chunk_id")
                if not chunk_id:
                    raise SystemExit("manifest missing chunk_id")
                src = scratch / "data/hpc/shards" / args.stage / chunk_id
                if not src.is_dir():
                    raise SystemExit(f"shard output missing: {src}")
                # Refuse to mark a chunk "done" without its stage's key artifact
                # (e.g. an ingest run that skipped every trial).
                required = {
                    "ingest": src / "trial_metadata_chunk.csv",
                    "preprocess": src / "signals_clean",
                    "features": src / "trial_features_chunk.parquet",
                }[args.stage]
                if not required.exists():
                    raise SystemExit(f"shard output incomplete, missing {required}")
                for f in sorted(src.rglob("*")):
                    if f.is_file():
                        tf.add(f, arcname=f.relative_to(scratch).as_posix())

        if args.stage == "anomaly":
            pid = manifest.get("held_out_participant_id", "fold")
            safe = str(pid).replace("/", "_")
            osd_dest = f"{osd_gg}/hpc/oof/anomaly/fold_{safe}.tar.gz"
            _push_osdf(tar_path, osd_dest)
            print(f"uploaded {tar_path} -> {osd_dest}")
            if args.error:
                return 1
            return 0

        chunk_id = manifest["chunk_id"]
        osd_dest = f"{osd_gg}/hpc/shards/{args.stage}/{chunk_id}.tar.gz"
        out_name = scratch / "shard_out.tar.gz"
        if out_name.exists():
            out_name.unlink(missing_ok=True)
        tar_path.rename(out_name)
        print(f"packaged {out_name} for condor upload -> {osd_dest}")
        if local_staging_enabled():
            local_dest = gaitguard_root() / "hpc" / "shards" / args.stage / f"{chunk_id}.tar.gz"
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_name, local_dest)
            print(f"local staging copy -> {local_dest}")
        if args.error:
            return 1
        return 0
    finally:
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
