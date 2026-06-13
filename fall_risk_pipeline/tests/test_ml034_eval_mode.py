"""ML-034: evaluation has a single nested LOSO path (no fast/paper modes)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_pipeline_config_has_no_evaluation_mode_key():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    assert "mode" not in cfg["models"]["evaluation"]


def test_main_has_no_fast_eval_cli_flag():
    source = (PIPELINE_ROOT / "main.py").read_text(encoding="utf-8")
    assert "--fast-eval" not in source
    assert "fast_eval" not in source


def test_evaluator_has_no_fast_switch():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(encoding="utf-8")
    assert "self.fast" not in source
    assert "fast: bool" not in source
    assert "_nested_group_evaluate_model(name, X, y, groups, cohorts)" in source
