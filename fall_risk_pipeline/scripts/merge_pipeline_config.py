#!/usr/bin/env python3
"""Deep-merge pipeline_config.local.yaml onto pipeline_config.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default="configs/pipeline_config.local.yaml")
    parser.add_argument("--out", default="configs/pipeline_config.local.generated.yaml")
    args = parser.parse_args()

    master_path = PIPELINE_ROOT / "configs/pipeline_config.yaml"
    local_path = PIPELINE_ROOT / args.local
    with open(master_path, encoding="utf-8") as f:
        master = yaml.safe_load(f)
    with open(local_path, encoding="utf-8") as f:
        local = yaml.safe_load(f) or {}

    merged = deep_merge(master, local)
    out = PIPELINE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
