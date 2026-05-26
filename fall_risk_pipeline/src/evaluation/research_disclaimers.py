"""
Non-clinical language and study-design limitations for research-prototype outputs.
"""

from __future__ import annotations

RESEARCH_PROTOTYPE_DISCLAIMER = (
    "Research prototype — not for clinical use without validation."
)

STUDY_DESIGN_LIMITATIONS: dict[str, str] = {
    "retrospective": (
        "Retrospective secondary analysis of existing recordings — "
        "no forward-in-time data collection for this project."
    ),
    "single_dataset": (
        "Single open dataset (Voisard et al., Figshare 10.6084/m9.figshare.28806086); "
        "no external multi-site replication reported."
    ),
    "no_prospective_follow_up": (
        "No prospective participant follow-up or incident-outcome ascertainment "
        "for this analysis."
    ),
    "no_ground_truth_fall_outcomes": (
        "No participant-level ground-truth fall outcomes; labels are cohort-level "
        "pathology tiers and literature-based fall-risk references only."
    ),
}

PROSPECTIVE_VALIDATION_PATH: list[str] = [
    "Enroll a new prospective cohort with pre-specified follow-up (e.g. 6–12 months).",
    "Collect adjudicated incident falls (and injurious falls) as primary endpoints.",
    "Validate frozen models at independent site(s) with outcome ascertainment blinded where feasible.",
    "Calibrate scores vs. Morse Fall Scale, STRATIFY, and observed fall rates.",
]

LIMITATIONS_BULLETS: list[str] = [
    RESEARCH_PROTOTYPE_DISCLAIMER,
    STUDY_DESIGN_LIMITATIONS["retrospective"],
    STUDY_DESIGN_LIMITATIONS["single_dataset"],
    STUDY_DESIGN_LIMITATIONS["no_prospective_follow_up"],
    STUDY_DESIGN_LIMITATIONS["no_ground_truth_fall_outcomes"],
    "Internal LOSO metrics on the same dataset — not independent prospective performance.",
    "Outputs are exploratory screening scores from open IMU gait data, not medical advice or a diagnosis.",
    "Single-trial API inference does not replicate multi-trial patient aggregation used in training.",
    "Risk thresholds (Youden J) are derived from internal cross-validation, not regulatory clearance.",
    "Do not use these results for treatment or fall-prevention decisions without prospective validation.",
]


def screening_note(elevated_prob: float, youden_prob: float) -> str:
    """Soft screening wording — never imperative clinical directives."""
    if elevated_prob >= youden_prob:
        return (
            "May warrant further clinical assessment "
            "(model score at/above internal validation cutoff; not a diagnosis)."
        )
    if elevated_prob >= 0.5 * youden_prob:
        return (
            "Borderline model score — interpret cautiously; "
            + RESEARCH_PROTOTYPE_DISCLAIMER.lower()
        )
    return (
        "Lower model score in this screening run; "
        + RESEARCH_PROTOTYPE_DISCLAIMER.lower()
    )


def limitations_payload() -> dict:
    return {
        "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
        "study_design": dict(STUDY_DESIGN_LIMITATIONS),
        "prospective_validation_path": list(PROSPECTIVE_VALIDATION_PATH),
        "points": list(LIMITATIONS_BULLETS),
    }
