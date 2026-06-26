"""Regression: subject-grouped splits wired before training (Klaver 2023 contrast)."""

from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def test_trainer_exports_subject_split_before_fit():
    source = (PIPELINE_ROOT / "src" / "models" / "trainer.py").read_text(encoding="utf-8")
    assert "ensure_subject_split_manifest" in source
    assert "DATA LEAKAGE" in (
        PIPELINE_ROOT / "src" / "dataset" / "subject_split.py"
    ).read_text(encoding="utf-8")


def test_evaluator_asserts_loso_disjoint():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(encoding="utf-8")
    assert "assert_loso_fold_disjoint" in source
    assert "ensure_subject_split_manifest" in source


def test_methods_documents_klaver_contrast():
    methods = (PIPELINE_ROOT.parent / "docs" / "paper" / "methods.md").read_text(encoding="utf-8")
    assert "Klaver" in methods
    assert "participant_id" in methods
    assert "70%" in methods
