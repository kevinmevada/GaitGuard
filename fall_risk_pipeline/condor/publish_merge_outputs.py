#!/usr/bin/env python3
"""Publish merged pipeline outputs from worker scratch to OSDF via stashcp."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def _push(local: Path, osd_dest: str) -> None:
    if not local.exists():
        return
    subprocess.run(["stashcp", str(local), osd_dest], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    osd_gg = os.environ.get("OSDF_GAITGUARD", "osdf:///ospool/ap40/data/kevin.mevada/gaitguard")
    proc = Path("data/processed")

    if args.stage == "ingest":
        _push(proc / "trial_metadata.csv", f"{osd_gg}/processed/trial_metadata.csv")
        signals = proc / "signals"
        if signals.is_dir():
            for f in signals.glob("*.parquet"):
                _push(f, f"{osd_gg}/processed/signals/{f.name}")
        ge = proc / "gait_events"
        if ge.is_dir():
            for f in ge.glob("*.csv"):
                _push(f, f"{osd_gg}/processed/gait_events/{f.name}")
    elif args.stage == "preprocess":
        clean = proc / "signals_clean"
        if clean.is_dir():
            for f in clean.glob("*.parquet"):
                _push(f, f"{osd_gg}/processed/signals_clean/{f.name}")
    else:
        feat = Path("data/features/trial_features.parquet")
        _push(feat, f"{osd_gg}/features/trial_features.parquet")

    out = Path("merge_out.tar.gz")
    marker = Path("merge_done.txt")
    marker.write_text(f"merge {args.stage}\n", encoding="utf-8")
    with tarfile.open(out, "w:gz") as tf:
        tf.add(marker, arcname=marker.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
