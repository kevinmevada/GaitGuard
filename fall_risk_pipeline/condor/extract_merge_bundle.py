#!/usr/bin/env python3
"""Extract merge bundle tarball on ap40 into canonical OSDF gaitguard paths."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

try:
    from local_paths import gaitguard_root
except ImportError:  # ap40 local universe without local_paths.py on path.
    import os

    def gaitguard_root() -> Path:
        raw = os.environ.get("OSDF_GAITGUARD", "/ospool/ap40/data/kevin.mevada/gaitguard")
        if raw.startswith("osdf://"):
            raw = raw[len("osdf://") :]
        return Path(raw)


def _gaitguard_root() -> Path:
    return gaitguard_root()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    args = parser.parse_args(argv)

    # Unpack into the canonical gaitguard tree (OSDF on ap40, .local_staging
    # locally) so the next stage's workers can stage these as inputs.
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
