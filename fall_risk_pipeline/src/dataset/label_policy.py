"""
Cohort → risk label policy (binary vs three-tier multiclass).

Multiclass tiers (pathology-based, from Voisard et al. cohort design):
  0 — Healthy (reference, ~5% annual fall rate)
  1 — Orthopedic elevated risk (HipOA, KneeOA, ACL; ~19–29% reference rates)
  2 — Neurological elevated risk (PD, CVA, CIPN, RIL; ~39–67%)

Binary collapse at threshold ≥1 merges tiers 1+2 and is **not** clinically equivalent
(HipOA 28.5% vs PD 67.3% fall probability). Use only with explicit justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES, COHORT_LABEL_MAP

MULTICLASS_NAMES = {
    0: "low (Healthy)",
    1: "moderate (orthopedic: HipOA/KneeOA/ACL)",
    2: "high (neurological: PD/CVA/CIPN/RIL)",
}

BINARY_STRATEGY_NOTES = {
    "threshold_ge_1": (
        "Legacy rule: binary high-risk if multiclass label ≥ 1. "
        "Groups orthopedic (e.g. HipOA ~28.5% fall prob.) with neurological "
        "(PD ~67.3%) pathology — clinically heterogeneous."
    ),
    "threshold_ge_2": (
        "Neurological-only high risk (multiclass ≥ 2). "
        "Separates orthopedic from neurological tiers; closer to "
        "pathology-stratified fall-risk literature (Lord et al. 1993; "
        "Allen et al. 2013 for PD/CVA-specific risk)."
    ),
}


@dataclass(frozen=True)
class ResolvedLabels:
    cohort: str
    multiclass_label: int
    training_label: int
    fall_probability: float
    label_mode: str
    binary_threshold: int | None


def get_dataset_label_config(config: dict) -> dict:
    ds = config.get("dataset", {})
    return {
        "label_mode": str(ds.get("label_mode", "multiclass")).lower(),
        "high_risk_threshold": int(ds.get("high_risk_threshold", 2)),
        "binary_strategy": str(ds.get("binary_strategy", "threshold_ge_2")),
        "cohort_labels": ds.get("cohort_labels", COHORT_LABEL_MAP),
    }


def multiclass_label_from_cohort(cohort: str, cohort_map: dict | None = None) -> int:
    mapping = cohort_map or COHORT_LABEL_MAP
    return int(mapping.get(str(cohort), 0))


def binary_label_from_multiclass(
    multiclass_label: int,
    threshold: int = 1,
) -> int:
    return int(multiclass_label >= threshold)


def resolve_labels(cohort: str, config: dict) -> ResolvedLabels:
    """Resolve training target and always retain multiclass tier."""
    cfg = get_dataset_label_config(config)
    mode = cfg["label_mode"]
    raw = multiclass_label_from_cohort(cohort, cfg.get("cohort_labels"))
    fp = COHORT_FALL_PROBABILITIES.get(str(cohort), 10.0) / 100.0

    if mode == "multiclass":
        training = raw
        threshold = None
    elif mode == "binary":
        threshold = cfg["high_risk_threshold"]
        training = binary_label_from_multiclass(raw, threshold)
    else:
        raise ValueError(
            f"Unknown label_mode '{mode}'. Use 'multiclass' or 'binary'."
        )

    return ResolvedLabels(
        cohort=str(cohort),
        multiclass_label=raw,
        training_label=int(training),
        fall_probability=float(fp),
        label_mode=mode,
        binary_threshold=threshold,
    )


def is_binary_task(y, config: dict | None = None) -> bool:
    cfg = get_dataset_label_config(config or {})
    if cfg["label_mode"] == "binary":
        return True
    classes = set(np_unique_labels(y))
    return classes <= {0, 1}


def np_unique_labels(y) -> list[int]:
    import numpy as np
    return sorted(int(v) for v in np.unique(np.asarray(y).astype(int)))


def n_classes_for_task(y, config: dict) -> int:
    import numpy as np
    if is_binary_task(y, config):
        return 2
    return max(2, len(np.unique(np.asarray(y).astype(int))))


def label_mode_description(config: dict) -> str:
    cfg = get_dataset_label_config(config)
    if cfg["label_mode"] == "multiclass":
        return (
            "Primary target: 3-class labels (0=Healthy, 1=orthopedic, 2=neurological). "
            "Preserves separation between OA/ACL and neurodegenerative/vascular cohorts."
        )
    strategy = cfg["binary_strategy"]
    note = BINARY_STRATEGY_NOTES.get(strategy, "")
    return (
        f"Primary target: binary (high-risk if multiclass ≥ {cfg['high_risk_threshold']}). "
        f"{note}"
    )


def sensitivity_binary_scenarios() -> list[dict[str, Any]]:
    """Alternative binary collapses for transparency tables (not default training)."""
    rows = []
    for threshold in (1, 2):
        rows.append({
            "high_risk_threshold": threshold,
            "rule": f"binary=1 if multiclass≥{threshold}",
            "strategy_key": f"threshold_ge_{threshold}",
            "clinical_note": BINARY_STRATEGY_NOTES.get(f"threshold_ge_{threshold}", ""),
            "cohorts_high_risk": [
                c for c, raw in COHORT_LABEL_MAP.items() if raw >= threshold
            ],
        })
    return rows
