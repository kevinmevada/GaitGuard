#!/usr/bin/env python3
"""Package merged pipeline outputs for HTCondor osdf upload (merge_out.tar.gz)."""

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


def _add_tree(tf: tarfile.TarFile, src: Path, arc_prefix: str) -> None:
    if not src.exists():
        return
    if src.is_file():
        tf.add(src, arcname=f"{arc_prefix}/{src.name}" if arc_prefix else src.name)
        return
    for f in sorted(src.rglob("*")):
        if f.is_file():
            rel = f.relative_to(src.parent if src.is_dir() else src.parent)
            tf.add(f, arcname=str(Path(arc_prefix) / rel).replace("\\", "/"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    scratch = Path.cwd()
    out = scratch / "merge_out.tar.gz"
    if out.exists():
        out.unlink()

    proc = Path("data/processed")
    with tarfile.open(out, "w:gz") as tf:
        if args.stage == "ingest":
            meta = proc / "trial_metadata.csv"
            if meta.is_file():
                tf.add(meta, arcname="processed/trial_metadata.csv")
            signals = proc / "signals"
            if signals.is_dir():
                for f in sorted(signals.glob("*.parquet")):
                    tf.add(f, arcname=f"processed/signals/{f.name}")
            ge = proc / "gait_events"
            if ge.is_dir():
                for f in sorted(ge.glob("*.csv")):
                    tf.add(f, arcname=f"processed/gait_events/{f.name}")
        elif args.stage == "preprocess":
            clean = proc / "signals_clean"
            if clean.is_dir():
                for f in sorted(clean.glob("*.parquet")):
                    tf.add(f, arcname=f"processed/signals_clean/{f.name}")
        else:
            for name in ("trial_features.parquet", "patient_features.parquet"):
                feat = Path("data/features") / name
                if feat.is_file():
                    tf.add(feat, arcname=f"features/{name}")

    print(f"packaged {out} ({out.stat().st_size} bytes)")
    if local_staging_enabled():
        bundle = gaitguard_root() / "hpc" / "merge_bundles" / f"{args.stage}.tar.gz"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, bundle)
        print(f"local merge bundle -> {bundle}")
    return 0 if out.is_file() and out.stat().st_size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
