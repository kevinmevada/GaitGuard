"""
Ordinal / continuous severity targets for Navita-style regression benchmarks.

Voisard does not ship per-participant UPDRS in the public ingest path. We use:
  1. ``ordinal_severity`` — pathology-tier proxy (HS=0, ortho=1, neuro=2)
  2. ``fall_probability_pct`` — cohort reference fall-risk % (continuous UPDRS proxy)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES, PATHOLOGY_KEY_MAP

# Navita-style pathology key → ordinal severity (user mapping; TKA ≈ KneeOA).
PATHOLOGY_ORDINAL_SEVERITY: dict[str, float] = {
    "HS": 0.0,
    "HOA": 1.0,
    "KOA": 1.0,
    "TKA": 1.0,
    "ACL": 1.0,
    "PD": 2.0,
    "CVA": 2.0,
    "MS": 2.0,
    "CIPN": 2.0,
    "RIL": 2.0,
}

COHORT_ORDINAL_SEVERITY: dict[str, float] = {
    "Healthy": 0.0,
    "HipOA": 1.0,
    "KneeOA": 1.0,
    "ACL": 1.0,
    "PD": 2.0,
    "CVA": 2.0,
    "CIPN": 2.0,
    "RIL": 2.0,
}

NEURO_COHORTS = frozenset({"PD", "CVA", "CIPN", "RIL"})
ORTHO_COHORTS = frozenset({"HipOA", "KneeOA", "ACL"})


def _severity_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("severity_regression") or {})


def ordinal_severity_from_cohort(cohort: str, config: dict[str, Any] | None = None) -> float:
    cfg = _severity_cfg(config or {})
    custom = cfg.get("ordinal_by_cohort") or {}
    if cohort in custom:
        return float(custom[cohort])
    return float(COHORT_ORDINAL_SEVERITY.get(str(cohort), float("nan")))


def fall_probability_target(cohort: str, fall_probability: float | None = None) -> float:
    """Continuous severity proxy on cohort fall-risk percent scale."""
    if fall_probability is not None and np.isfinite(fall_probability):
        return float(fall_probability)
    return float(COHORT_FALL_PROBABILITIES.get(str(cohort), float("nan")))


def attach_severity_targets(meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``severity_ordinal`` and ``severity_fall_probability_pct`` columns."""
    out = meta.copy()
    fp = out["fall_probability"] if "fall_probability" in out.columns else None
    out["severity_ordinal"] = [
        ordinal_severity_from_cohort(str(c), config) for c in out["cohort"]
    ]
    if fp is not None:
        out["severity_fall_probability_pct"] = fp.astype(float).values
    else:
        out["severity_fall_probability_pct"] = [
            fall_probability_target(str(c)) for c in out["cohort"]
        ]
    return out


def resolve_regression_target(
    row: pd.Series,
    target_name: str,
    config: dict[str, Any],
) -> float:
    if target_name == "fall_probability_pct":
        if "fall_probability" in row and pd.notna(row["fall_probability"]):
            return float(row["fall_probability"])
        return fall_probability_target(str(row["cohort"]))
    return ordinal_severity_from_cohort(str(row["cohort"]), config)


def cohort_in_scope(cohort: str, scope: str) -> bool:
    c = str(cohort)
    if scope == "all_8":
        return True
    if scope == "neuro_ortho":
        return c in NEURO_COHORTS or c in ORTHO_COHORTS
    if scope == "neuro_only":
        return c in NEURO_COHORTS
    if scope == "ortho_only":
        return c in ORTHO_COHORTS
    return True


def pathology_key_reference() -> dict[str, str]:
    return dict(PATHOLOGY_KEY_MAP)
