#!/usr/bin/env python3
"""Regenerate docs/paper/results.md from fall_risk_pipeline metrics (PUB-001)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.paper_results_sync import sync_paper_results  # noqa: E402


def main() -> int:
    config_path = PIPELINE_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    out = sync_paper_results(config)
    print(f"Paper results -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
