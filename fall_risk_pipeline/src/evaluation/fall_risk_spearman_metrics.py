"""
Spearman correlation — literature fall probability vs BiLSTM-AE anomaly score.

Clinical validation: anomaly score as a proxy for fall risk. Participant-level
mean LOSO anomaly scores are correlated with cohort reference fall probabilities
(Voisard / Lord et al. literature rates) or trial ``fall_probability`` metadata.

Within a single cohort, fall probability is constant (cohort label), so
``within_cohort`` ρ is not defined. Per-cohort manuscript rows use
``healthy_vs_{cohort}`` (Healthy + pathological tier) where fall probability varies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES
from src.evaluation.per_cohort_loso_metrics import HEALTHY, PATHOLOGICAL_COHORT_ORDER
from src.utils.progress import progress_bar

MIN_SPEARMAN_N = 3


def to_fall_probability_pct(values: pd.Series) -> pd.Series:
    """Normalize fall probability to percent scale (5.2–67.3)."""
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().any() and float(s.median()) <= 1.0:
        return s * 100.0
    return s


def fall_probability_pct_for_cohort(cohort: str, fall_probability: float | None = None) -> float:
    if fall_probability is not None and np.isfinite(fall_probability):
        fp = float(fall_probability)
        return fp * 100.0 if fp <= 1.0 else fp
    return float(COHORT_FALL_PROBABILITIES.get(str(cohort), float("nan")))


def attach_fall_probability_pct(trial_df: pd.DataFrame) -> pd.DataFrame:
    out = trial_df.copy()
    if "fall_probability" in out.columns and out["fall_probability"].notna().any():
        out["fall_probability_pct"] = to_fall_probability_pct(out["fall_probability"])
    else:
        out["fall_probability_pct"] = [
            fall_probability_pct_for_cohort(str(c)) for c in out["cohort"]
        ]
    return out


def aggregate_participant_means(trial_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """One row per participant: mean anomaly score + fall probability."""
    required = {"participant_id", "cohort", score_col}
    missing = required - set(trial_df.columns)
    if missing:
        raise ValueError(f"Trial frame missing columns: {missing}")

    enriched = attach_fall_probability_pct(trial_df)
    grouped = (
        enriched.groupby(["participant_id", "cohort"], as_index=False)
        .agg(
            mean_anomaly_score=(score_col, "mean"),
            n_trials=(score_col, "count"),
            fall_probability_pct=("fall_probability_pct", "first"),
        )
    )
    return grouped


def spearman_fall_risk_vs_score(
    fall_probability_pct: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    min_n: int = MIN_SPEARMAN_N,
) -> dict[str, Any]:
    x = np.asarray(fall_probability_pct, dtype=float)
    y = np.asarray(anomaly_scores, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "n_unique_fall_prob": int(len(np.unique(x[mask]))) if n else 0,
            "defined": False,
            "note": f"insufficient n (need ≥{min_n})",
        }
    x_m, y_m = x[mask], y[mask]
    n_unique_fp = int(len(np.unique(x_m)))
    if n_unique_fp < 2:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "n_unique_fall_prob": n_unique_fp,
            "defined": False,
            "note": "fall probability constant within comparison set",
        }
    rho, p = spearmanr(x_m, y_m)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": n,
        "n_unique_fall_prob": n_unique_fp,
        "defined": True,
        "note": "",
    }


def _row_from_spearman(
    *,
    comparison_scope: str,
    cohort: str,
    participant_df: pd.DataFrame,
    stats: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    return {
        "comparison_scope": comparison_scope,
        "cohort": cohort,
        "level": level,
        "n_participants": int(len(participant_df)),
        "n_trials": int(participant_df["n_trials"].sum()) if "n_trials" in participant_df.columns else int("nan"),
        "spearman_rho": stats["spearman_rho"],
        "p_value": stats["p_value"],
        "n": stats["n"],
        "n_unique_fall_prob": stats["n_unique_fall_prob"],
        "defined": stats["defined"],
        "note": stats.get("note", ""),
        "mean_fall_probability_pct": float(participant_df["fall_probability_pct"].mean()),
        "mean_anomaly_score": float(participant_df["mean_anomaly_score"].mean()),
        "fall_probability_source": "trial_metadata_or_cohort_reference",
    }


def compute_fall_risk_spearman_table(
    trial_df: pd.DataFrame,
    score_col: str,
    *,
    min_n: int = MIN_SPEARMAN_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (summary_table, participant_level_frame).

    Summary rows:
      - global_all_participants
      - healthy_vs_{cohort} for each pathological tier
      - within_{cohort} (documented NA when fall prob is cohort-constant)
    """
    participants = aggregate_participant_means(trial_df, score_col)
    rows: list[dict[str, Any]] = []

    global_stats = spearman_fall_risk_vs_score(
        participants["fall_probability_pct"].values,
        participants["mean_anomaly_score"].values,
        min_n=min_n,
    )
    rows.append(
        _row_from_spearman(
            comparison_scope="global_all_participants",
            cohort="ALL",
            participant_df=participants,
            stats=global_stats,
            level="participant_mean",
        )
    )

    for cohort in progress_bar(
        PATHOLOGICAL_COHORT_ORDER, desc="fall_risk_spearman", unit="cohort"
    ):
        mask = participants["cohort"].isin([HEALTHY, cohort])
        subset = participants.loc[mask].copy()
        if subset.empty:
            continue
        stats = spearman_fall_risk_vs_score(
            subset["fall_probability_pct"].values,
            subset["mean_anomaly_score"].values,
            min_n=min_n,
        )
        rows.append(
            _row_from_spearman(
                comparison_scope=f"healthy_vs_{cohort}",
                cohort=cohort,
                participant_df=subset,
                stats=stats,
                level="participant_mean",
            )
        )

    all_cohorts = [HEALTHY, *PATHOLOGICAL_COHORT_ORDER]
    for cohort in progress_bar(
        all_cohorts, desc="fall_risk_spearman within", unit="cohort"
    ):
        subset = participants.loc[participants["cohort"] == cohort]
        if subset.empty:
            continue
        stats = spearman_fall_risk_vs_score(
            subset["fall_probability_pct"].values,
            subset["mean_anomaly_score"].values,
            min_n=min_n,
        )
        rows.append(
            _row_from_spearman(
                comparison_scope=f"within_{cohort}",
                cohort=cohort,
                participant_df=subset,
                stats=stats,
                level="participant_mean",
            )
        )

    trial_enriched = attach_fall_probability_pct(trial_df)
    trial_stats = spearman_fall_risk_vs_score(
        trial_enriched["fall_probability_pct"].values,
        trial_enriched[score_col].astype(float).values,
        min_n=min_n,
    )
    rows.append(
        {
            "comparison_scope": "global_all_trials",
            "cohort": "ALL",
            "level": "trial",
            "n_participants": int(participants["participant_id"].nunique()),
            "n_trials": int(len(trial_enriched)),
            "spearman_rho": trial_stats["spearman_rho"],
            "p_value": trial_stats["p_value"],
            "n": trial_stats["n"],
            "n_unique_fall_prob": trial_stats["n_unique_fall_prob"],
            "defined": trial_stats["defined"],
            "note": trial_stats.get("note", ""),
            "mean_fall_probability_pct": float(trial_enriched["fall_probability_pct"].mean()),
            "mean_anomaly_score": float(trial_enriched[score_col].astype(float).mean()),
            "fall_probability_source": "trial_metadata_or_cohort_reference",
        }
    )

    return pd.DataFrame(rows), participants
