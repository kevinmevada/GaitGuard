"""Generate Section 2 methodological novelty comparison table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.reporting.novelty_comparison import write_novelty_artifacts


def run_novelty_comparison_table(config: dict) -> dict[str, Any]:
    cfg = config.get("novelty_table") or {}
    if not cfg.get("enabled", True):
        logger.info("Novelty comparison table disabled")
        return {}

    metrics_dir = Path(config["paths"]["metrics"])
    paper_dir: Path | None = None
    if cfg.get("sync_paper_docs", True):
        cfg_path = Path((config.get("_pipeline_meta") or {}).get("config_path", "configs/pipeline_config.yaml"))
        pipeline_root = cfg_path.resolve().parent.parent
        candidate = pipeline_root.parent / "docs" / "paper"
        if candidate.parent.is_dir():
            paper_dir = candidate
        else:
            paper_dir = pipeline_root / "docs" / "paper"

    summary = write_novelty_artifacts(metrics_dir, paper_docs_dir=paper_dir)
    logger.info(
        "Novelty comparison table → {} (GaitGuard sole full-firsts: {})",
        metrics_dir / "novelty_comparison_table.md",
        summary.get("gaitguard_unique_full_firsts"),
    )
    return summary
