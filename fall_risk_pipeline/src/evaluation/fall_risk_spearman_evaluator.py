"""
Clinical validation — Spearman ρ (fall probability vs BiLSTM-AE anomaly score).

Requires ``bilstm_ae_loso_oof_scores.csv`` from the anomaly stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.evaluation.fall_risk_spearman_metrics import compute_fall_risk_spearman_table
from src.evaluation.per_cohort_loso_metrics import PATHOLOGICAL_COHORT_ORDER, cohort_display_name
from src.models.bilstm_ae_scoring import METHOD_ENSEMBLE

DEFAULT_SCORE_COL = f"{METHOD_ENSEMBLE}_score"


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("fall_risk_spearman") or {}


def _load_oof(metrics_dir: Path, score_col: str) -> pd.DataFrame:
    path = metrics_dir / "bilstm_ae_loso_oof_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} — run `python main.py --stage anomaly` (BiLSTM-AE LOSO) first."
        )
    df = pd.read_csv(path)
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col!r} not in {path.name}")
    required = {"cohort", "participant_id", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"OOF file missing columns: {required - set(df.columns)}")
    return df


def _merge_fall_probability(trial_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if "fall_probability" in trial_df.columns and trial_df["fall_probability"].notna().any():
        return trial_df
    paths = config.get("paths") or {}
    feat_dir = Path(paths.get("features", "data/features"))
    feat_path = feat_dir / "trial_features.parquet"
    if not feat_path.is_file():
        return trial_df
    meta = pd.read_parquet(feat_path)
    cols = [c for c in ("trial_id", "fall_probability", "cohort") if c in meta.columns]
    if "trial_id" not in cols or "trial_id" not in trial_df.columns:
        return trial_df
    merged = trial_df.merge(meta[cols], on="trial_id", how="left", suffixes=("", "_meta"))
    if "fall_probability_meta" in merged.columns:
        merged["fall_probability"] = merged["fall_probability"].fillna(merged["fall_probability_meta"])
        merged = merged.drop(columns=["fall_probability_meta"])
    if "cohort_meta" in merged.columns:
        merged["cohort"] = merged["cohort"].fillna(merged["cohort_meta"])
        merged = merged.drop(columns=["cohort_meta"])
    return merged


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not pd.notna(v):
        return "—"
    return f"{float(v):.{nd}f}"


def render_fall_risk_spearman_md(summary_df: pd.DataFrame, *, score_col: str) -> str:
    global_row = summary_df.loc[summary_df["comparison_scope"] == "global_all_participants"]
    per_cohort = summary_df[
        summary_df["comparison_scope"].str.startswith("healthy_vs_", na=False)
    ].copy()

    lines = [
        "# Fall-risk clinical validation — Spearman correlation",
        "",
        "Anomaly score is evaluated as a **proxy for fall risk** by correlating "
        "literature / metadata fall probability with BiLSTM-AE LOSO anomaly scores.",
        "",
        f"**Score column:** `{score_col}`  ",
        "**Fall probability:** cohort reference rates (Voisard / clinical literature) "
        "or trial `fall_probability` from ingest metadata when present.",
        "",
        "## Primary result — all participants",
        "",
    ]

    if not global_row.empty:
        g = global_row.iloc[0]
        sig = "significant" if pd.notna(g["p_value"]) and float(g["p_value"]) < 0.05 else "not significant"
        lines.extend(
            [
                f"- **Spearman ρ** = {_fmt(g['spearman_rho'])} (p = {_fmt(g['p_value'])}, "
                f"n = {int(g['n'])} participants, {sig} at α = 0.05)",
                f"- Mean anomaly score = {_fmt(g['mean_anomaly_score'])}; "
                f"mean reference fall probability = {_fmt(g['mean_fall_probability_pct'], 1)}%",
                "",
            ]
        )
    else:
        lines.append("_Global participant-level correlation not computed._\n")

    lines.extend(
        [
            "## Per pathological cohort (Healthy vs cohort contrast)",
            "",
            "Within a single cohort, fall probability is cohort-constant, so ρ is reported "
            "for **Healthy + pathological tier** participants where fall probability varies.",
            "",
            "| Cohort | vs Healthy | n participants | ρ | p-value | Mean score | Ref. fall prob. (%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in per_cohort.itertuples(index=False):
        cohort = str(row.cohort)
        display = cohort_display_name(cohort)
        rho = _fmt(row.spearman_rho) if row.defined else "NA"
        pval = _fmt(row.p_value) if row.defined else "—"
        lines.append(
            f"| **{display}** | Healthy + {display} | {int(row.n_participants)} | "
            f"{rho} | {pval} | {_fmt(row.mean_anomaly_score)} | "
            f"{_fmt(row.mean_fall_probability_pct, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Positive ρ** — higher literature fall-risk cohorts show higher anomaly scores "
            "(supports deployment as clinical decision-support signal).",
            "- **AUROC/F1** answer discrimination; this table answers **clinical relevance**.",
            "- Rows `within_{cohort}` in the CSV are NA by design (no fall-probability variance).",
            "",
            "### Cohort reference fall probabilities (%)",
            "",
            "| Cohort | Reference fall prob. (%) |",
            "|---|---:|",
        ]
    )
    from src.ingestion.data_loader import COHORT_FALL_PROBABILITIES

    for cohort in ("Healthy", *PATHOLOGICAL_COHORT_ORDER):
        fp = COHORT_FALL_PROBABILITIES.get(cohort, float("nan"))
        lines.append(f"| {cohort_display_name(cohort)} | {_fmt(fp, 1)} |")

    lines.append("")
    return "\n".join(lines)


def run_fall_risk_spearman_correlation(config: dict) -> pd.DataFrame:
    cfg = _cfg(config)
    if not cfg.get("enabled", True):
        logger.info("Fall-risk Spearman correlation disabled")
        return pd.DataFrame()

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    score_col = str(cfg.get("score_column", DEFAULT_SCORE_COL))
    min_n = int(cfg.get("min_n", 3))

    oof = _load_oof(metrics_dir, score_col)
    oof = _merge_fall_probability(oof, config)

    summary_df, participant_df = compute_fall_risk_spearman_table(
        oof, score_col, min_n=min_n
    )

    csv_path = metrics_dir / "fall_risk_spearman_correlation.csv"
    summary_df.to_csv(csv_path, index=False)
    participant_df.to_csv(metrics_dir / "fall_risk_spearman_participant_means.csv", index=False)

    global_row = summary_df.loc[summary_df["comparison_scope"] == "global_all_participants"]
    payload = {
        "score_column": score_col,
        "global_participant_spearman": (
            global_row.iloc[0].to_dict() if not global_row.empty else {}
        ),
        "per_cohort_healthy_contrast": summary_df[
            summary_df["comparison_scope"].str.startswith("healthy_vs_", na=False)
        ].to_dict(orient="records"),
        "manuscript_guidance": (
            "Report global_all_participants ρ as the primary clinical validation statistic. "
            "Per-cohort healthy_vs_* rows show pathology-tier alignment with fall-risk literature."
        ),
    }
    (metrics_dir / "fall_risk_spearman_correlation.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    md_body = render_fall_risk_spearman_md(summary_df, score_col=score_col)
    (metrics_dir / "fall_risk_spearman_correlation.md").write_text(md_body, encoding="utf-8")

    if cfg.get("sync_paper_docs", True):
        cfg_path = Path((config.get("_pipeline_meta") or {}).get("config_path", "configs/pipeline_config.yaml"))
        pipeline_root = cfg_path.resolve().parent.parent
        paper_dir = pipeline_root.parent / "docs" / "paper"
        if paper_dir.parent.is_dir():
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "fall_risk_spearman_correlation.md").write_text(md_body, encoding="utf-8")

    g = global_row.iloc[0] if not global_row.empty else None
    logger.info(
        "Fall-risk Spearman correlation → {} (global ρ={:.4f}, p={:.4g})",
        csv_path,
        float(g["spearman_rho"]) if g is not None and pd.notna(g["spearman_rho"]) else float("nan"),
        float(g["p_value"]) if g is not None and pd.notna(g["p_value"]) else float("nan"),
    )
    return summary_df
