"""Pipeline stage ordering and full-run connectivity."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_main_stages_match_canonical_list():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from main import STAGES
    from src.utils.pipeline_stages import PIPELINE_STAGES

    assert STAGES == list(PIPELINE_STAGES)


def test_anomaly_runs_before_post_anomaly_stages():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.utils.pipeline_stages import PIPELINE_STAGES, POST_ANOMALY_STAGES

    anomaly_idx = PIPELINE_STAGES.index("anomaly")
    for stage in POST_ANOMALY_STAGES:
        if stage in PIPELINE_STAGES:
            assert PIPELINE_STAGES.index(stage) > anomaly_idx, (
                f"{stage} must run after anomaly"
            )


def test_resolve_stages_all_and_comma_list():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.utils.pipeline_stages import PIPELINE_STAGES, resolve_stages

    assert resolve_stages("all") == list(PIPELINE_STAGES)
    subset = resolve_stages("anomaly,report,fall_risk_spearman")
    assert subset == ["anomaly", "fall_risk_spearman", "report"]


def test_main_has_progress_stop_start_for_substage_bars():
    source = (PIPELINE_ROOT / "main.py").read_text(encoding="utf-8")
    assert "progress.stop()" in source
    assert "progress.start()" in source
