"""
Aggregate discriminative metrics across classical, DL, and GaitGuard primary rows.

Writes ``competitor_discriminative_metrics.csv`` + ``competitor_discriminative_metrics.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.evaluation.competitor_metrics import DISCRIMINATIVE_COLUMNS
from src.evaluation.dl_baseline_evaluator import DISPLAY_NAMES as DL_DISPLAY
from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE

METRIC_COLS = [
    "f1_weighted",
    "balanced_accuracy",
    "mcc",
    "auroc",
    "sensitivity",
    "specificity",
    "precision",
    "cohen_kappa",
]

DISPLAY_COLS = [
    "Model",
    "Paradigm",
    "F1 (weighted)",
    "Balanced Acc.",
    "MCC",
    "AUROC",
    "Sensitivity",
    "Specificity",
    "Precision",
    "Cohen κ",
]


def _fmt(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{float(v):.4f}"


def _row_from_classical(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            {
                "model": str(r.model),
                "display_name": str(r.model).replace("_", " ").title(),
                "paradigm": "classical_paradigm_1",
                "source": "classical_baseline_metrics.csv",
                "f1_weighted": getattr(r, "f1_weighted", getattr(r, "f1", float("nan"))),
                "balanced_accuracy": r.balanced_accuracy,
                "mcc": getattr(r, "mcc", float("nan")),
                "auroc": getattr(r, "auroc", getattr(r, "auc", float("nan"))),
                "sensitivity": getattr(r, "sensitivity", float("nan")),
                "specificity": getattr(r, "specificity", float("nan")),
                "precision": getattr(r, "precision", float("nan")),
                "cohen_kappa": getattr(r, "cohen_kappa", float("nan")),
            }
        )
    return rows


def _row_from_dl(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for r in df.itertuples(index=False):
        name = getattr(r, "display_name", str(r.model))
        rows.append(
            {
                "model": str(r.model),
                "display_name": name,
                "paradigm": getattr(r, "paradigm", "competitor_paradigm_2_dl"),
                "source": "dl_baseline_metrics.csv",
                "f1_weighted": getattr(r, "f1_weighted", getattr(r, "f1", float("nan"))),
                "balanced_accuracy": r.balanced_accuracy,
                "mcc": getattr(r, "mcc", float("nan")),
                "auroc": getattr(r, "auroc", float("nan")),
                "sensitivity": getattr(r, "sensitivity", float("nan")),
                "specificity": getattr(r, "specificity", float("nan")),
                "precision": getattr(r, "precision", float("nan")),
                "cohen_kappa": getattr(r, "cohen_kappa", float("nan")),
            }
        )
    return rows


def _row_from_bilstm(metrics_dir: Path) -> dict[str, Any] | None:
    path = metrics_dir / "bilstm_ae_anomaly_metrics.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    ens = df[df["method"] == ENDPOINT_BILSTM_AE_ENSEMBLE]
    if ens.empty:
        return None
    r = ens.iloc[0]
    return {
        "model": ENDPOINT_BILSTM_AE_ENSEMBLE,
        "display_name": "BiLSTM-AE (GaitGuard)",
        "paradigm": "gaitguard_primary",
        "source": "bilstm_ae_anomaly_metrics.csv",
        "f1_weighted": float(r.get("f1_weighted", r.get("f1", float("nan")))),
        "balanced_accuracy": float(r.get("balanced_accuracy", float("nan"))),
        "mcc": float(r.get("mcc", float("nan"))),
        "auroc": float(r.get("auc", float("nan"))),
        "sensitivity": float(r.get("sensitivity", float("nan"))),
        "specificity": float(r.get("specificity", float("nan"))),
        "precision": float(r.get("precision", float("nan"))),
        "cohen_kappa": float(r.get("cohen_kappa", float("nan"))),
    }


def run_competitor_discriminative_matrix(config: dict) -> pd.DataFrame:
    cfg = (config.get("competitor_metrics") or {})
    if not cfg.get("enabled", True):
        return pd.DataFrame()

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    cl_path = metrics_dir / "classical_baseline_metrics.csv"
    if cl_path.is_file():
        rows.extend(_row_from_classical(pd.read_csv(cl_path)))

    dl_path = metrics_dir / "dl_baseline_metrics.csv"
    if dl_path.is_file():
        rows.extend(_row_from_dl(pd.read_csv(dl_path)))

    bilstm = _row_from_bilstm(metrics_dir)
    if bilstm and not any(r["model"] == ENDPOINT_BILSTM_AE_ENSEMBLE for r in rows):
        rows.append(bilstm)

    if not rows:
        logger.warning("No baseline metrics found for competitor discriminative matrix")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "competitor_discriminative_metrics.csv"
    df.to_csv(csv_path, index=False)

    gaitguard = df[df["model"] == ENDPOINT_BILSTM_AE_ENSEMBLE]
    mcc_lead = float("nan")
    if not gaitguard.empty:
        mcc_lead = float(gaitguard.iloc[0].get("mcc", float("nan")))

    threshold = float(cfg.get("mcc_abstract_lead_threshold", 0.7))
    summary = {
        "n_models": int(len(df)),
        "gaitguard_mcc": mcc_lead,
        "mcc_abstract_lead_threshold": threshold,
        "abstract_headline": (
            "mcc_primary"
            if pd.notna(mcc_lead) and mcc_lead >= threshold
            else "auroc_primary"
        ),
        "metric_columns": list(METRIC_COLS),
    }
    (metrics_dir / "competitor_discriminative_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _write_markdown(metrics_dir / "competitor_discriminative_metrics.md", df, summary)
    logger.info("Competitor discriminative matrix → {} ({} models)", csv_path, len(df))
    return df


def _write_markdown(path: Path, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Core discriminative metrics — competitor matrix",
        "",
        "F1 (weighted), balanced accuracy, MCC, AUROC, sensitivity, specificity, "
        "precision, Cohen's κ — LOSO out-of-fold.",
        "",
    ]
    if summary.get("abstract_headline") == "mcc_primary":
        lines.append(
            f"**Abstract lead metric:** MCC = {summary.get('gaitguard_mcc', float('nan')):.4f} "
            f"(≥ {summary.get('mcc_abstract_lead_threshold', 0.7):.1f} threshold)."
        )
        lines.append("")
    else:
        lines.append(
            "**Abstract lead metric:** AUROC (threshold-independent headline); report MCC for rigor."
        )
        lines.append("")

    header = "| " + " | ".join(DISPLAY_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(DISPLAY_COLS)) + "|"
    lines.extend([header, sep])

    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.display_name),
                    str(row.paradigm),
                    _fmt(row.f1_weighted),
                    _fmt(row.balanced_accuracy),
                    _fmt(row.mcc),
                    _fmt(row.auroc),
                    _fmt(row.sensitivity),
                    _fmt(row.specificity),
                    _fmt(row.precision),
                    _fmt(getattr(row, "cohen_kappa", float("nan"))),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
