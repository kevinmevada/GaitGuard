"""
Label and class-balance reporting for the fall-risk cohort mapping.

Multiclass tiers (default training target when label_mode=multiclass):
  Healthy→0 | HipOA, KneeOA, ACL→1 | PD, CVA, CIPN, RIL→2

Binary risk_label (label_mode=binary): multiclass tier >= high_risk_threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight

from src.dataset.label_policy import (
    MULTICLASS_NAMES,
    binary_label_from_multiclass,
    get_dataset_label_config,
    label_mode_description,
    multiclass_label_from_cohort,
    sensitivity_binary_scenarios,
)
from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES, COHORT_LABEL_MAP

BINARY_CLASS_NAMES = {0: "low_risk (Healthy)", 1: "high_risk (binary positive)"}


def raw_label_from_cohort(cohort: str) -> int:
    return multiclass_label_from_cohort(cohort)


def balanced_scale_pos_weight(y: np.ndarray) -> float:
    """
    XGBoost scale_pos_weight for binary positive class: n_negative / n_positive.
    Returns 1.0 when not a two-class problem.
    """
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if len(classes) != 2:
        return 1.0
    counts = np.bincount(y, minlength=int(classes.max()) + 1)
    pos = int(classes.max())
    neg = int(classes.min())
    if counts[pos] == 0:
        return 1.0
    return float(counts[neg] / counts[pos])


def sklearn_balanced_weights(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def summarize_patient_labels(
    df: pd.DataFrame,
    config: dict | None = None,
) -> dict[str, Any]:
    """Summarize participant-level cohort and training-label counts."""
    cfg = get_dataset_label_config(config or {})
    label_mode = cfg["label_mode"]

    if "participant_id" in df.columns:
        work = df.drop_duplicates("participant_id").copy()
        level = "participant"
    else:
        work = df.copy()
        level = "row"

    n_total = len(work)
    cohort_counts = (
        work["cohort"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if "cohort" in work.columns
        else {}
    )

    if "multiclass_label" in work.columns:
        multi_counts = (
            work["multiclass_label"].value_counts().sort_index().astype(int).to_dict()
        )
        multi_counts = {int(k): int(v) for k, v in multi_counts.items()}
    elif "cohort" in work.columns:
        work = work.assign(
            multiclass_label=work["cohort"].map(
                lambda c: multiclass_label_from_cohort(str(c))
            )
        )
        multi_counts = (
            work["multiclass_label"].value_counts().sort_index().astype(int).to_dict()
        )
        multi_counts = {int(k): int(v) for k, v in multi_counts.items()}
    else:
        multi_counts = {}

    if "risk_label" in work.columns:
        training_counts = (
            work["risk_label"].value_counts().sort_index().astype(int).to_dict()
        )
        training_counts = {int(k): int(v) for k, v in training_counts.items()}
    else:
        training_counts = {}

    y = work["risk_label"].values.astype(int) if "risk_label" in work.columns else np.array([])
    imbalance_ratio = float("nan")
    if len(y) > 0 and len(np.unique(y)) == 2:
        counts = np.bincount(y)
        imbalance_ratio = float(counts[0] / counts[1]) if counts[1] > 0 else float("nan")

    return {
        "level": level,
        "n_total": n_total,
        "label_mode": label_mode,
        "label_mode_description": label_mode_description(config or {}),
        "cohort_counts": {str(k): int(v) for k, v in cohort_counts.items()},
        "multiclass_counts": multi_counts,
        "training_label_counts": training_counts,
        "binary_class_names": BINARY_CLASS_NAMES,
        "multiclass_names": MULTICLASS_NAMES,
        "cohort_fall_probability_pct": COHORT_FALL_PROBABILITIES,
        "label_mapping": COHORT_LABEL_MAP,
        "high_risk_threshold": cfg["high_risk_threshold"],
        "binary_strategy": cfg["binary_strategy"],
        "imbalance_ratio_neg_to_pos": imbalance_ratio,
        "balanced_scale_pos_weight": balanced_scale_pos_weight(y) if len(y) else float("nan"),
        "sklearn_balanced_weights": sklearn_balanced_weights(y) if len(y) else {},
    }


def cohort_label_table(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Per-cohort counts with multiclass tier, training label, and binary sensitivities."""
    if "cohort" not in df.columns:
        return pd.DataFrame()

    cfg = get_dataset_label_config(config or {})
    threshold = cfg["high_risk_threshold"]

    if "participant_id" in df.columns:
        work = df.drop_duplicates("participant_id")
    else:
        work = df

    rows = []
    for cohort, grp in work.groupby("cohort", dropna=False):
        raw = raw_label_from_cohort(str(cohort))
        if "multiclass_label" in grp.columns:
            raw = int(grp["multiclass_label"].iloc[0])
        training = (
            int(grp["risk_label"].iloc[0])
            if "risk_label" in grp.columns
            else raw if cfg["label_mode"] == "multiclass" else binary_label_from_multiclass(raw, threshold)
        )
        rows.append({
            "cohort": str(cohort),
            "n_participants": int(len(grp)),
            "multiclass_label": raw,
            "multiclass_name": MULTICLASS_NAMES.get(raw, "unknown"),
            "training_label": training,
            "binary_at_threshold_1": binary_label_from_multiclass(raw, 1),
            "binary_at_threshold_2": binary_label_from_multiclass(raw, 2),
            "reference_fall_probability_pct": COHORT_FALL_PROBABILITIES.get(str(cohort), float("nan")),
        })
    return pd.DataFrame(rows).sort_values("n_participants", ascending=False)


def save_class_distribution_reports(
    df: pd.DataFrame,
    metrics_dir: Path,
    *,
    prefix: str = "",
    config: dict | None = None,
) -> dict[str, Any]:
    """Write CSV + markdown summaries; return summary dict."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_patient_labels(df, config)
    cohort_df = cohort_label_table(df, config)

    tag = f"{prefix}_" if prefix else ""
    cohort_path = metrics_dir / f"{tag}class_distribution_by_cohort.csv"
    summary_path = metrics_dir / f"{tag}class_distribution_summary.json"

    if not cohort_df.empty:
        cohort_df.to_csv(cohort_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    sens_path = metrics_dir / f"{tag}binary_label_sensitivity.csv"
    pd.DataFrame(sensitivity_binary_scenarios()).to_csv(sens_path, index=False)

    mode = summary["label_mode"]
    md_lines = [
        "# Class distribution report",
        "",
        f"**Level:** {summary['level']} (N={summary['n_total']})",
        f"**Label mode:** `{mode}` — {summary['label_mode_description']}",
        "",
        "## Cohort → label mapping",
        "",
        "| Cohort | Multiclass | Train label | Binary (≥1) | Binary (≥2) | Ref. fall % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not cohort_df.empty:
        for row in cohort_df.itertuples(index=False):
            md_lines.append(
                f"| {row.cohort} | {int(row.multiclass_label)} | {int(row.training_label)} | "
                f"{int(row.binary_at_threshold_1)} | {int(row.binary_at_threshold_2)} | "
                f"{float(row.reference_fall_probability_pct):.1f} |"
            )
    else:
        for cohort, raw in sorted(COHORT_LABEL_MAP.items(), key=lambda x: x[1]):
            fp = COHORT_FALL_PROBABILITIES.get(cohort, float("nan"))
            n = summary["cohort_counts"].get(cohort, 0)
            md_lines.append(
                f"| {cohort} | {raw} | — | {1 if raw >= 1 else 0} | {1 if raw >= 2 else 0} | {fp} | ({n})"
            )

    md_lines.extend([
        "",
        f"## Training label counts (`label_mode={mode}`)",
        "",
    ])
    for cls, count in sorted(summary["training_label_counts"].items()):
        if mode == "multiclass":
            name = summary["multiclass_names"].get(cls, str(cls))
        else:
            name = summary["binary_class_names"].get(cls, str(cls))
        pct = 100.0 * count / max(summary["n_total"], 1)
        md_lines.append(f"- **{name}** (label={cls}): **{count}** ({pct:.1f}%)")

    if summary["multiclass_counts"] and mode == "binary":
        md_lines.extend(["", "## Underlying multiclass tier counts", ""])
        for cls, count in sorted(summary["multiclass_counts"].items()):
            name = summary["multiclass_names"].get(cls, str(cls))
            md_lines.append(f"- {name}: **{count}**")

    if np.isfinite(summary["imbalance_ratio_neg_to_pos"]):
        md_lines.extend([
            "",
            f"**Binary imbalance ratio** (neg/pos): {summary['imbalance_ratio_neg_to_pos']:.2f}",
            f"**XGBoost scale_pos_weight** (binary only): {summary['balanced_scale_pos_weight']:.3f}",
        ])

    md_lines.extend([
        "",
        "Alternative binary collapses: see `binary_label_sensitivity.csv` and `docs/label_binning.md`.",
    ])

    md_path = metrics_dir / f"{tag}class_distribution_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info(
        f"Class distribution ({mode}): N={summary['n_total']} | "
        f"labels={summary['training_label_counts']}"
    )
    return summary
