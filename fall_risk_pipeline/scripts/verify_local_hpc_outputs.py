#!/usr/bin/env python3
"""Verify expected artifacts after local sharded HPC run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _check_file(path: Path, label: str, errors: list[str]) -> None:
    if path.is_file() and path.stat().st_size > 0:
        print(f"OK  {label} ({path.stat().st_size} bytes)")
    else:
        errors.append(f"MISSING or empty: {label} ({path})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gaitguard-root",
        default=str(PIPELINE_ROOT / ".local_staging" / "gaitguard"),
        help="Canonical output tree (OSDF equivalent) where merge bundles are extracted",
    )
    args = parser.parse_args()

    # Final canonical artifacts land in the gaitguard root (OSDF on ap40,
    # .local_staging/gaitguard locally) via extract_merge_bundle.
    root = Path(args.gaitguard_root)
    proc = root / "processed"
    feat = root / "features"
    errors: list[str] = []

    _check_file(proc / "trial_metadata.csv", "trial_metadata", errors)
    for label, sub in (("signals", "signals"), ("signals_clean", "signals_clean")):
        files = sorted((proc / sub).glob("*.parquet"))
        if files:
            _check_file(files[0], f"{label} parquet sample", errors)
        else:
            errors.append(f"MISSING: {label} under {proc / sub}")
    _check_file(feat / "trial_features.parquet", "trial_features", errors)
    _check_file(feat / "patient_features.parquet", "patient_features", errors)

    if (proc / "trial_metadata.csv").is_file():
        meta = pd.read_csv(proc / "trial_metadata.csv")
        print(f"OK  trials in metadata: {len(meta)}")

    if errors:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\nAll required HPC outputs present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
