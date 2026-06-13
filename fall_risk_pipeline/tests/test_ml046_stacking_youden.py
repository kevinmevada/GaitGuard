"""ML-046: stacking ensemble Youden must use inner grouped OOF, not in-sample train scores."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = REPO_ROOT / "fall_risk_pipeline" / "src" / "evaluation" / "evaluator.py"


def _ensemble_one_subject_source() -> str:
    tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_evaluate_ensemble_one_subject":
            return ast.get_source_segment(
                EVALUATOR_PATH.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError("_evaluate_ensemble_one_subject not found")


def test_stacking_uses_inner_oof_threshold():
    source = _ensemble_one_subject_source()
    assert "_stacking_inner_oof_threshold" in source
    assert "train_prob = predict_ensemble_oof_proba" not in source


def test_stacking_inner_oof_helper_exists():
    text = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert "def _stacking_inner_oof_threshold" in text
