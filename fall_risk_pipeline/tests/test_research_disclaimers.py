"""Soft language for research prototype outputs."""

from src.evaluation.research_disclaimers import (
    RESEARCH_PROTOTYPE_DISCLAIMER,
    STUDY_DESIGN_LIMITATIONS,
    limitations_payload,
    screening_note,
)


def test_disclaimer_wording():
    assert "not for clinical use" in RESEARCH_PROTOTYPE_DISCLAIMER.lower()


def test_screening_note_high_uses_soft_language():
    text = screening_note(0.8, 0.5)
    assert "May warrant further clinical assessment" in text
    assert "Immediate" not in text
    assert "intervention" not in text.lower()


def test_screening_note_low_includes_disclaimer():
    text = screening_note(0.1, 0.5)
    assert "research prototype" in text.lower()


def test_study_design_limitations_present():
    assert "retrospective" in STUDY_DESIGN_LIMITATIONS
    assert "fall outcomes" in STUDY_DESIGN_LIMITATIONS["no_ground_truth_fall_outcomes"].lower()


def test_limitations_payload_structure():
    payload = limitations_payload()
    assert "study_design" in payload
    assert "prospective_validation_path" in payload
    assert len(payload["prospective_validation_path"]) >= 3
