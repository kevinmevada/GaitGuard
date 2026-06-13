"""ML-048: split-protocol comparison export naming."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_evaluator_writes_split_protocol_comparison_csv():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "split_protocol_comparison.csv" in source
    assert "Subject-leakage comparison saved" not in source


def test_reporter_reads_split_protocol_with_legacy_fallback():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "reporter.py").read_text(
        encoding="utf-8"
    )
    assert "_split_protocol_comparison_section" in source
    assert "split_protocol_comparison.csv" in source
    assert "leakage_comparison.csv" in source
    assert "ML-048" in source
