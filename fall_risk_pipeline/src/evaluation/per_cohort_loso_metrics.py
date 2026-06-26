"""
Per-cohort LOSO metrics — one-vs-healthy screening per pathology tier.

Do not aggregate away cohort-specific clinical signal. Each pathological cohort
is evaluated separately against Healthy under pooled LOSO OOF scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.clinical_threshold import youden_threshold
from src.evaluation.competitor_metrics import compute_discriminative_metrics
from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES

HEALTHY = "Healthy"

# Manuscript-facing order (Voisard pathology keys; MS → CIPN/RIL neuropathy cohorts).
PATHOLOGICAL_COHORT_ORDER: tuple[str, ...] = (
    "PD",
    "CVA",
    "CIPN",
    "RIL",
    "HipOA",
    "KneeOA",
    "ACL",
)

COHORT_DISPLAY_ALIASES: dict[str, str] = {
    "HipOA": "HOA",
    "KneeOA": "TKA",
    "CIPN": "CIPN",
    "RIL": "RIL",
}

ALL_COHORT_ORDER: tuple[str, ...] = (HEALTHY, *PATHOLOGICAL_COHORT_ORDER)


def cohort_display_name(cohort: str) -> str:
    return COHORT_DISPLAY_ALIASES.get(cohort, cohort)


def _participant_counts(participant_ids: np.ndarray, mask: np.ndarray) -> int:
    return int(len(np.unique(participant_ids[mask])))


def one_vs_healthy_metrics(
    cohorts: np.ndarray,
    y_score: np.ndarray,
    target_cohort: str,
    *,
    threshold: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Binary screening: target cohort (positive) vs Healthy (negative)."""
    c = np.asarray(cohorts, dtype=str)
    mask = (c == target_cohort) | (c == HEALTHY)
    if int((c == target_cohort).sum()) == 0:
        return {}

    y_true = (c[mask] == target_cohort).astype(int)
    scores = np.asarray(y_score, dtype=float)[mask]
    if len(np.unique(y_true)) < 2:
        return {}

    thr = float(threshold) if threshold is not None else youden_threshold(y_true, scores)
    y_pred = (scores >= thr).astype(int)
    disc = compute_discriminative_metrics(y_true, y_pred, y_score=scores, config=config)
    path_mask = c[mask] == target_cohort
    healthy_mask = c[mask] == HEALTHY

    return {
        "cohort": target_cohort,
        "cohort_display": cohort_display_name(target_cohort),
        "comparison": f"{cohort_display_name(target_cohort)} vs Healthy",
        "n_trials_pathological": int(path_mask.sum()),
        "n_trials_healthy": int(healthy_mask.sum()),
        "n_trials_total": int(mask.sum()),
        "auroc": disc["auroc"],
        "f1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_weighted": disc["f1_weighted"],
        "mcc": disc["mcc"],
        "sensitivity": disc["sensitivity"],
        "specificity": disc["specificity"],
        "balanced_accuracy": disc["balanced_accuracy"],
        "threshold_youden": thr,
        "mean_score_pathological": float(np.mean(scores[path_mask])),
        "mean_score_healthy": float(np.mean(scores[healthy_mask])),
        "anomaly_rate_pct": float(100.0 * np.mean(y_pred[path_mask])),
        "healthy_flag_rate_pct": float(100.0 * np.mean(y_pred[healthy_mask])),
        "reference_fall_probability_pct": float(COHORT_FALL_PROBABILITIES.get(target_cohort, float("nan"))),
        "score_gap_vs_healthy": float(np.mean(scores[path_mask]) - np.mean(scores[healthy_mask])),
    }


def cohort_score_summary(
    cohorts: np.ndarray,
    y_score: np.ndarray,
    participant_ids: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = np.asarray(cohorts, dtype=str)
    scores = np.asarray(y_score, dtype=float)
    pids = np.asarray(participant_ids, dtype=str)
    for cohort in ALL_COHORT_ORDER:
        mask = c == cohort
        if not mask.any():
            continue
        flagged = scores[mask] >= threshold
        rows.append(
            {
                "cohort": cohort,
                "cohort_display": cohort_display_name(cohort) if cohort != HEALTHY else HEALTHY,
                "n_trials": int(mask.sum()),
                "n_participants": _participant_counts(pids, mask),
                "mean_anomaly_score": float(np.mean(scores[mask])),
                "median_anomaly_score": float(np.median(scores[mask])),
                "std_anomaly_score": float(np.std(scores[mask])),
                "anomaly_rate_pct": float(100.0 * np.mean(flagged)),
                "reference_fall_probability_pct": float(
                    COHORT_FALL_PROBABILITIES.get(cohort, float("nan"))
                ),
            }
        )
    return rows


def kruskal_wallis_across_cohorts(
    cohorts: np.ndarray,
    y_score: np.ndarray,
    *,
    cohort_order: tuple[str, ...] = ALL_COHORT_ORDER,
) -> dict[str, Any]:
    """Kruskal-Wallis H-test on trial-level anomaly scores across cohorts."""
    c = np.asarray(cohorts, dtype=str)
    scores = np.asarray(y_score, dtype=float)
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for cohort in cohort_order:
        mask = c == cohort
        if int(mask.sum()) < 2:
            continue
        groups.append(scores[mask])
        labels.append(cohort)

    if len(groups) < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_groups": len(groups),
            "cohorts": labels,
            "significant_at_0_05": False,
        }

    stat, p = kruskal(*groups)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": len(groups),
        "cohorts": labels,
        "significant_at_0_05": bool(p < 0.05),
        "test": "kruskal_wallis_trial_level_scores",
    }


def kruskal_wallis_participant_means(
    cohorts: np.ndarray,
    y_score: np.ndarray,
    participant_ids: np.ndarray,
    *,
    cohort_order: tuple[str, ...] = ALL_COHORT_ORDER,
) -> dict[str, Any]:
    """KW on participant-mean scores (one value per participant)."""
    df = pd.DataFrame(
        {
            "cohort": np.asarray(cohorts, dtype=str),
            "score": np.asarray(y_score, dtype=float),
            "participant_id": np.asarray(participant_ids, dtype=str),
        }
    )
    part = df.groupby(["participant_id", "cohort"], as_index=False)["score"].mean()
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for cohort in cohort_order:
        sub = part.loc[part["cohort"] == cohort, "score"].values
        if len(sub) < 2:
            continue
        groups.append(sub.astype(float))
        labels.append(cohort)

    if len(groups) < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_groups": len(groups),
            "cohorts": labels,
            "significant_at_0_05": False,
        }

    stat, p = kruskal(*groups)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": len(groups),
        "cohorts": labels,
        "significant_at_0_05": bool(p < 0.05),
        "test": "kruskal_wallis_participant_mean_scores",
    }


def pd_clinical_paradox_note(pd_row: dict[str, Any], cohort_rows: list[dict[str, Any]]) -> str | None:
    """
    PD often shows low anomaly rate despite highest reference fall probability —
    internally consistent pathological gait that stays near the healthy manifold.
    """
    if pd_row.get("cohort") != "PD":
        return None
    rates = [r["anomaly_rate_pct"] for r in cohort_rows if r.get("cohort") != HEALTHY]
    if not rates:
        return None
    pd_rate = float(pd_row.get("anomaly_rate_pct", float("nan")))
    pd_fp = float(pd_row.get("reference_fall_probability_pct", float("nan")))
    if not np.isfinite(pd_rate) or not np.isfinite(pd_fp):
        return None
    if pd_rate <= float(np.median(rates)) and pd_fp >= 55.0:
        return (
            "**PD clinical paradox (discuss explicitly):** Parkinson's disease carries the "
            f"highest reference fall probability in this dataset ({pd_fp:.1f}%), yet LOSO "
            f"anomaly flagging is comparatively low ({pd_rate:.1f}% of PD trials above "
            "the Youden threshold). This pattern is clinically meaningful — PD gait can be "
            "pathologically impaired yet **internally consistent** (narrow, stereotyped "
            "kinematics), producing modest reconstruction/latent deviation from a healthy "
            "manifold. Averaging PD into a single pooled metric would hide this dissociation "
            "between epidemiological fall risk and unsupervised anomaly score."
        )
    return None
