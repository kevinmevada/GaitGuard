#!/usr/bin/env python3
"""Sync canonical Front_end assets into api/static (STR-001)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "Front_end"
DST = REPO_ROOT / "api" / "static"
ASSETS = ("index.html", "main.js", "style.css")


def main() -> int:
    if not SRC.is_dir():
        print(f"Missing source directory: {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        src_file = SRC / name
        if not src_file.is_file():
            print(f"Missing asset: {src_file}", file=sys.stderr)
            return 1
        shutil.copy2(src_file, DST / name)
        print(f"OK  {src_file.relative_to(REPO_ROOT)} -> {DST.relative_to(REPO_ROOT)}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
