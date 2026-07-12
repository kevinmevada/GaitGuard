"""
Per-cohort LOSO results — detailed pathology-tier breakdown (Results centerpiece).

Reports AUROC and F1 **per pathological cohort** (one-vs-Healthy), Kruskal-Wallis
across cohort anomaly scores, and explicit PD clinical discussion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.clinical_threshold import youden_threshold
from src.evaluation.per_cohort_loso_metrics import (
    ALL_COHORT_ORDER,
    HEALTHY,
    PATHOLOGICAL_COHORT_ORDER,
    cohort_display_name,
    cohort_score_summary,
    kruskal_wallis_across_cohorts,
    kruskal_wallis_participant_means,
    one_vs_healthy_metrics,
    pd_clinical_paradox_note,
)
from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE
from src.models.bilstm_ae_scoring import METHOD_ENSEMBLE
from src.utils.progress import progress_bar

DEFAULT_SCORE_COL = f"{METHOD_ENSEMBLE}_score"


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("per_cohort_loso") or {}


def _load_oof_frame(metrics_dir: Path, score_col: str) -> pd.DataFrame:
    path = metrics_dir / "bilstm_ae_loso_oof_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} — run `python main.py --stage anomaly` (BiLSTM-AE LOSO) first."
        )
    df = pd.read_csv(path)
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col!r} not in {path.name}; columns={list(df.columns)}")
    required = {"cohort", "participant_id", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"OOF file missing columns: {required - set(df.columns)}")
    return df


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{float(v):.{nd}f}"


def render_per_cohort_results_md(
    metrics_df: pd.DataFrame,
    score_summary_df: pd.DataFrame,
    kw_trial: dict[str, Any],
    kw_participant: dict[str, Any],
    *,
    clinical_notes: list[str],
    model: str,
    score_col: str,
    global_threshold: float,
) -> str:
    lines = [
        "# Per-cohort LOSO results — pathology-tier screening (detailed)",
        "",
        "This section reports **cohort-resolved** LOSO out-of-fold performance. "
        "Pooled means are supplementary only; clinical heterogeneity across the eight "
        "Voisard cohorts is the primary result.",
        "",
        f"**Model:** {model} (`{score_col}`)  ",
        f"**Global Youden threshold (all pathological vs Healthy):** {_fmt(global_threshold)}",
        "",
        "> Voisard does not include an MS-labelled cohort. Neuropathy-tier signal is carried "
        "by **CIPN** (chemotherapy-induced peripheral neuropathy) and **RIL** (radiculopathy/leg "
        "pain). Orthopedic cohorts map to manuscript aliases **HOA** (HipOA) and **TKA** (KneeOA).",
        "",
        "## 1. One-vs-Healthy screening per pathological cohort",
        "",
        "Each row is a **separate** binary task: cohort *c* (positive) vs Healthy (negative). "
        "AUROC and F1 are **not** macro-averaged across cohorts.",
        "",
        "| Cohort | vs Healthy | n trials (path.) | n participants | AUROC | F1 | MCC | Sens. | Spec. | "
        "Anomaly rate (%) | Ref. fall prob. (%) | Mean score gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    any_suppressed = False
    for row in metrics_df.itertuples(index=False):
        n_participants = int(getattr(row, "n_participants_pathological", row.n_trials_pathological))
        is_suppressed = getattr(row, "auc_status", "stable") == "unstable_small_n"
        any_suppressed = any_suppressed or is_suppressed
        cohort_label = f"{row.cohort_display} †" if is_suppressed else row.cohort_display
        lines.append(
            f"| **{cohort_label}** | {row.comparison} | {int(row.n_trials_pathological)} | "
            f"{n_participants} | "
            f"{_fmt(row.auroc)} | {_fmt(row.f1_binary)} | {_fmt(row.mcc)} | "
            f"{_fmt(row.sensitivity)} | {_fmt(row.specificity)} | "
            f"{_fmt(row.anomaly_rate_pct, 1)} | {_fmt(row.reference_fall_probability_pct, 1)} | "
            f"{_fmt(row.score_gap_vs_healthy)} |"
        )

    if any_suppressed:
        min_n_note = (
            "† AUROC/F1/MCC/sensitivity/specificity suppressed (`auc_status: unstable_small_n`): "
            "cohort has fewer participants than `models.evaluation.cohort_auc_min_n` "
            "(default 25) and cannot support a stable point estimate. Do not cite these cells; "
            "see `docs/paper/methods.md` §10 for the rule this mirrors."
        )
        lines.append("")
        lines.append(min_n_note)

    lines.extend(
        [
            "",
            "### Interpretation guide",
            "",
            "- **AUROC / F1** — discrimination for that cohort only; compare across rows, do not average.",
            "- **Anomaly rate** — % of pathological trials flagged at the cohort-specific Youden threshold "
            "(re-fit on Healthy + that cohort's OOF trials).",
            "- **Ref. fall prob.** — literature reference fall-risk percentage for the cohort label "
            "(not a prospective outcome in this dataset).",
            "- **Mean score gap** — pathological minus healthy mean anomaly score on the same comparison set.",
            "",
            "## 2. Anomaly score distribution by cohort (all eight cohorts)",
            "",
            "| Cohort | n trials | n participants | Mean score | Median | SD | Anomaly rate (%) | Ref. fall prob. (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in score_summary_df.itertuples(index=False):
        bold = "**" if str(row.cohort) == "PD" else ""
        end = "**" if bold else ""
        lines.append(
            f"| {bold}{row.cohort_display}{end} | {int(row.n_trials)} | {int(row.n_participants)} | "
            f"{_fmt(row.mean_anomaly_score)} | {_fmt(row.median_anomaly_score)} | "
            f"{_fmt(row.std_anomaly_score)} | {_fmt(row.anomaly_rate_pct, 1)} | "
            f"{_fmt(row.reference_fall_probability_pct, 1)} |"
        )

    lines.extend(
        [
            "",
            "## 3. Kruskal-Wallis — cohort differences in anomaly score",
            "",
            "Tests whether anomaly-score distributions differ across cohorts "
            "(non-parametric; trial-level and participant-mean variants).",
            "",
            f"- **Trial-level scores:** H = {_fmt(kw_trial.get('statistic'), 3)}, "
            f"p = {_fmt(kw_trial.get('p_value'), 4)} "
            f"({'significant' if kw_trial.get('significant_at_0_05') else 'not significant'} at α=0.05), "
            f"k = {kw_trial.get('n_groups', '—')} cohorts",
            f"- **Participant-mean scores:** H = {_fmt(kw_participant.get('statistic'), 3)}, "
            f"p = {_fmt(kw_participant.get('p_value'), 4)} "
            f"({'significant' if kw_participant.get('significant_at_0_05') else 'not significant'} at α=0.05), "
            f"k = {kw_participant.get('n_groups', '—')} cohorts",
            "",
            "## 4. Clinical discussion — do not average away cohort signal",
            "",
        ]
    )

    if clinical_notes:
        for note in clinical_notes:
            lines.append(note)
            lines.append("")
    else:
        lines.append(
            "_No auto-generated PD paradox note (thresholds not met). Inspect per-cohort "
            "anomaly rates vs reference fall probabilities manually._"
        )
        lines.append("")

    lines.extend(
        [
            "The eight-cohort Voisard design enables contrasts that single-disease studies "
            "cannot replicate: high fall-probability neurological cohorts (PD, CVA) vs "
            "orthopedic mechanical gait (HOA, TKA, ACL) vs neuropathy-tier cohorts (CIPN, RIL). "
            "Report each row in the main text; reserve pooled AUROC for supplementary material only.",
            "",
        ]
    )
    return "\n".join(lines)


def run_per_cohort_loso_results(config: dict) -> pd.DataFrame:
    cfg = _cfg(config)
    if not cfg.get("enabled", True):
        logger.info("Per-cohort LOSO reporting disabled")
        return pd.DataFrame()

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    score_col = str(cfg.get("score_column", DEFAULT_SCORE_COL))
    model = str(cfg.get("model", ENDPOINT_BILSTM_AE_ENSEMBLE))

    oof = _load_oof_frame(metrics_dir, score_col)
    cohorts = oof["cohort"].astype(str).values
    scores = oof[score_col].astype(float).values
    pids = oof["participant_id"].astype(str).values

    # Global threshold: all pathological vs healthy (for score summary flag rates).
    global_mask = np.isin(cohorts, list(PATHOLOGICAL_COHORT_ORDER)) | (cohorts == HEALTHY)
    y_global = np.isin(cohorts[global_mask], list(PATHOLOGICAL_COHORT_ORDER)).astype(int)
    global_threshold = youden_threshold(y_global, scores[global_mask])

    metric_rows: list[dict[str, Any]] = []
    for cohort in progress_bar(
        PATHOLOGICAL_COHORT_ORDER, desc="per_cohort_loso", unit="cohort"
    ):
        row = one_vs_healthy_metrics(cohorts, scores, cohort, config=config)
        if not row:
            logger.warning("Skipping per-cohort metrics — no trials for {}", cohort)
            continue
        mask = cohorts == cohort
        row["n_participants_pathological"] = int(len(np.unique(pids[mask])))
        row["model"] = model
        row["score_column"] = score_col

        min_n = int(
            config.get("models", {}).get("evaluation", {}).get("cohort_auc_min_n", 25)
        )
        unstable = row["n_participants_pathological"] < min_n
        row["auc_status"] = "unstable_small_n" if unstable else "stable"
        if unstable:
            # Mirror evaluator.py's cohort-level suppression rule (ML-044/MED-003):
            # do not report point-estimate discriminative metrics for cohorts too
            # small to support a stable AUROC/F1/MCC estimate.
            for suppressed_field in ("auroc", "f1_binary", "f1_weighted", "mcc", "sensitivity", "specificity", "balanced_accuracy"):
                row[suppressed_field] = float("nan")

        metric_rows.append(row)

    if not metric_rows:
        logger.warning("No per-cohort LOSO rows produced")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metric_rows)
    score_summary = cohort_score_summary(cohorts, scores, pids, threshold=global_threshold)
    score_summary_df = pd.DataFrame(score_summary)

    kw_trial = kruskal_wallis_across_cohorts(cohorts, scores, cohort_order=ALL_COHORT_ORDER)
    kw_participant = kruskal_wallis_participant_means(
        cohorts, scores, pids, cohort_order=ALL_COHORT_ORDER
    )

    pd_row = next((r for r in metric_rows if r["cohort"] == "PD"), None)
    clinical_notes: list[str] = []
    if pd_row:
        note = pd_clinical_paradox_note(pd_row, metric_rows)
        if note:
            clinical_notes.append(note)

    csv_path = metrics_dir / "per_cohort_loso_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    score_summary_df.to_csv(metrics_dir / "per_cohort_anomaly_score_summary.csv", index=False)
    oof.to_csv(metrics_dir / "per_cohort_loso_trial_scores.csv", index=False)

    kw_payload = {
        "trial_level": kw_trial,
        "participant_mean": kw_participant,
        "global_threshold_youden": global_threshold,
        "clinical_notes": clinical_notes,
        "cohort_display_aliases": {
            c: cohort_display_name(c) for c in PATHOLOGICAL_COHORT_ORDER
        },
    }
    (metrics_dir / "per_cohort_kruskal_wallis.json").write_text(
        json.dumps(kw_payload, indent=2),
        encoding="utf-8",
    )

    md_body = render_per_cohort_results_md(
        metrics_df,
        score_summary_df,
        kw_trial,
        kw_participant,
        clinical_notes=clinical_notes,
        model=model,
        score_col=score_col,
        global_threshold=global_threshold,
    )
    (metrics_dir / "per_cohort_loso_results.md").write_text(md_body, encoding="utf-8")

    if cfg.get("sync_paper_docs", True):
        cfg_path = Path((config.get("_pipeline_meta") or {}).get("config_path", "configs/pipeline_config.yaml"))
        pipeline_root = cfg_path.resolve().parent.parent
        paper_dir = pipeline_root.parent / "docs" / "paper"
        if paper_dir.parent.is_dir():
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "per_cohort_loso_results.md").write_text(md_body, encoding="utf-8")

    logger.info(
        "Per-cohort LOSO results → {} ({} pathological cohorts; KW p={:.4g})",
        csv_path,
        len(metrics_df),
        kw_trial.get("p_value", float("nan")),
    )
    return metrics_df
