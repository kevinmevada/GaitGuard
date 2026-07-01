#!/usr/bin/env python3
"""Extract merge bundle tarball on ap40 into canonical OSDF gaitguard paths."""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


def _gaitguard_root() -> Path:
    raw = os.environ.get("OSDF_GAITGUARD", "/ospool/ap40/data/kevin.mevada/gaitguard")
    if raw.startswith("osdf://"):
        raw = raw[len("osdf://") :]
    return Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)

    root = _gaitguard_root()
    bundle = root / "hpc" / "merge_bundles" / f"{args.stage}.tar.gz"
    if not bundle.is_file():
        print(f"merge bundle missing: {bundle}", file=sys.stderr)
        return 1

    print(f"extracting {bundle} -> {root}")
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(root)
    print("extract done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
