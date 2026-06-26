"""Tests for Section 2 novelty comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_gaitguard_sole_row_with_all_firsts():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.reporting.novelty_comparison import (
        GAITGUARD_STUDY,
        gaitguard_unique_full_firsts,
        novelty_dataframe,
    )

    df = novelty_dataframe()
    assert gaitguard_unique_full_firsts(df)
    gg = df[df["study"] == GAITGUARD_STUDY].iloc[0]
    assert bool(gg["strict_loso"]) is True
    assert bool(gg["one_class_ensemble"]) is True
    assert bool(gg["cross_dataset_eval"]) is True
    assert int(gg["cohort_count"]) == 8


def test_write_novelty_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.novelty_table_evaluator import run_novelty_comparison_table
    from src.reporting.novelty_comparison import render_novelty_markdown

    metrics_dir = tmp_path / "metrics"
    config = {
        "paths": {"metrics": str(metrics_dir)},
        "_pipeline_meta": {
            "config_path": str(PIPELINE_ROOT / "configs" / "pipeline_config.yaml"),
        },
        "novelty_table": {"enabled": True, "sync_paper_docs": True},
    }
    summary = run_novelty_comparison_table(config)

    assert summary["gaitguard_unique_full_firsts"] is True
    assert (metrics_dir / "novelty_comparison_table.csv").is_file()
    assert (metrics_dir / "novelty_comparison_table.md").is_file()
    paper_path = REPO_ROOT / "docs" / "paper" / "table1_novelty_comparison.md"
    assert paper_path.is_file()

    md = render_novelty_markdown()
    assert "BiLSTM-AE" in md
    assert "DAPHNET" in md
    assert "✓" in md

    loaded = json.loads((metrics_dir / "novelty_comparison_summary.json").read_text())
    assert len(loaded["three_firsts"]) == 3
