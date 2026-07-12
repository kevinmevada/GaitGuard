#!/usr/bin/env python3
"""
GaitGuard — run the full pipeline on your PC.

Usage:
    python run_local.py
    python run_local.py --stage evaluate
    python run_local.py --config fall_risk_pipeline/configs/pipeline_config.yaml
    python run_local.py --seed-data --trials 6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GaitGuard on your local machine.")
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Pipeline YAML (relative to fall_risk_pipeline/)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        help="Stage name, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--seed-data",
        action="store_true",
        help="Seed synthetic raw IMU data before running (smoke tests)",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument(
        "--use-local-config",
        action="store_true",
        help="Merge configs/pipeline_config.local.yaml onto the master config",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("GAITGUARD_FORCE_PROGRESS", "1")
    os.chdir(PIPELINE_ROOT)

    config = args.config
    if args.use_local_config:
        result = subprocess.run(
            [sys.executable, "scripts/merge_pipeline_config.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        config = result.stdout.strip()

    if args.seed_data:
        cmd = [sys.executable, "scripts/seed_local_raw_data.py"]
        if args.trials is not None:
            cmd.extend(["--trials", str(args.trials)])
        if args.rows is not None:
            cmd.extend(["--rows", str(args.rows)])
        subprocess.run(cmd, check=True)

    subprocess.run(
        [sys.executable, "main.py", "--config", config, "--stage", args.stage],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
