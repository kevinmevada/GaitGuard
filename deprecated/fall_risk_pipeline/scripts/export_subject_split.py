#!/usr/bin/env python3
"""Export subject-grouped train/val/test manifest (70/15/15 HS + patho test-only)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

import yaml  # noqa: E402

from src.dataset.subject_split import ensure_subject_split_manifest  # noqa: E402


def main() -> None:
    cfg_path = PIPELINE_ROOT / "configs" / "pipeline_config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for key, rel in config.get("paths", {}).items():
        if isinstance(rel, str) and not Path(rel).is_absolute():
            config["paths"][key] = str(PIPELINE_ROOT / rel)

    split = ensure_subject_split_manifest(config)
    print(
        f"Subject split: train={len(split.train_ids)} "
        f"val={len(split.val_ids)} test={len(split.test_ids)}"
    )


if __name__ == "__main__":
    main()
