"""
Statistical benchmark rigor — Wilcoxon + Holm vs BiLSTM-AE, Critical Difference diagram.

Loads per-model LOSO OOF scores, builds leave-one-participant-out jackknife AUROC
vectors (paired across models), then:
  1. Wilcoxon signed-rank vs GaitGuard reference + Holm correction
  2. Friedman + Nemenyi Critical Difference diagram (UCR benchmark standard)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.critical_difference import (
    build_cd_summary,
    plot_critical_difference_diagram,
    wilcoxon_vs_reference,
)
from src.evaluation.loso_oof_scores import (
    align_jackknife_aurocs,
    discover_oof_models,
    leave_one_participant_out_aurocs,
    load_bilstm_ae_oof,
    load_model_oof,
    oof_scores_dir,
)
from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE

DISPLAY_NAMES: dict[str, str] = {
    ENDPOINT_BILSTM_AE_ENSEMBLE: "BiLSTM-AE (GaitGuard)",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "logistic_regression_l2": "Logistic Regression (L2)",
    "logistic_regression_l1": "Logistic Regression (L1)",
    "knn": "k-NN",
    "minirocket": "MINIROCKET",
    "rocket": "ROCKET",
    "inception_time": "InceptionTime",
    "deep_conv_lstm": "DeepConvLSTM",
}


def _stat_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("statistical_benchmark") or {}


def _collect_jackknife_aurocs(
    metrics_dir: Path,
    models: list[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    model_aucs: dict[str, tuple[np.ndarray, list[str]]] = {}
    for model in models:
        if model == ENDPOINT_BILSTM_AE_ENSEMBLE:
            df = load_bilstm_ae_oof(metrics_dir)
        else:
            df = load_model_oof(metrics_dir, model)
        if df is None or df.empty:
            logger.warning("No OOF scores for model {}", model)
            continue
        aucs, pids = leave_one_participant_out_aurocs(
            df["y_true"].values,
            df["score"].values,
            df["participant_id"].values,
        )
        if len(aucs) < 3:
            logger.warning("Insufficient jackknife folds for {} (n={})", model, len(aucs))
            continue
        model_aucs[model] = (aucs, pids)

    return align_jackknife_aurocs(model_aucs)


def run_statistical_benchmark(config: dict) -> dict[str, Any]:
    scfg = _stat_cfg(config)
    if not scfg.get("enabled", True):
        logger.info("Statistical benchmark disabled in config")
        return {}

    metrics_dir = Path(config["paths"]["metrics"])
    figures_dir = Path(config.get("paths", {}).get("figures_models", "results/figures/models"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference = str(scfg.get("reference_model", ENDPOINT_BILSTM_AE_ENSEMBLE))
    alpha = float(scfg.get("alpha", 0.05))
    models = list(scfg.get("models") or discover_oof_models(metrics_dir, reference))
    if reference not in models:
        if load_bilstm_ae_oof(metrics_dir) is not None:
            models.append(reference)
    models = sorted(set(models))

    if len(models) < 2:
        logger.warning(
            "Statistical benchmark needs ≥2 models with OOF scores in {}; "
            "run classical_baselines, dl_baselines, and anomaly stages first",
            oof_scores_dir(metrics_dir),
        )
        return {}

    pids, aligned = _collect_jackknife_aurocs(metrics_dir, models)
    if len(pids) < 3 or len(aligned) < 2:
        logger.warning("Insufficient aligned jackknife AUROC folds for statistical tests")
        return {}

    model_names = [m for m in models if m in aligned]
    cd_summary = build_cd_summary(aligned, model_names, alpha=alpha)

    wilcoxon_df = wilcoxon_vs_reference(aligned, reference, alpha=alpha)
    wilcoxon_path = metrics_dir / "wilcoxon_vs_bilstm_ae.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)

    # Jackknife fold table for reproducibility
    jackknife_rows: list[dict[str, Any]] = []
    for i, pid in enumerate(pids):
        row: dict[str, Any] = {"participant_id": pid, "jackknife_fold": i}
        for model in model_names:
            row[f"auroc_{model}"] = float(aligned[model][i])
        jackknife_rows.append(row)
    jackknife_df = pd.DataFrame(jackknife_rows)
    jackknife_df.to_csv(metrics_dir / "jackknife_auroc_by_fold.csv", index=False)

    cd_fig = plot_critical_difference_diagram(
        cd_summary["average_ranks"],
        float(cd_summary["critical_difference"]),
        figures_dir / "critical_difference_auroc.png",
        title="Critical Difference — LOSO jackknife AUROC ranks",
        display_names=DISPLAY_NAMES,
        alpha=alpha,
    )

    summary = {
        "reference_model": reference,
        "alpha": alpha,
        "n_jackknife_folds": len(pids),
        "n_models": len(model_names),
        "models": model_names,
        "friedman": cd_summary["friedman"],
        "average_ranks": cd_summary["average_ranks"],
        "critical_difference": cd_summary["critical_difference"],
        "wilcoxon_vs_reference": wilcoxon_df.to_dict(orient="records"),
        "artifacts": {
            "wilcoxon_csv": str(wilcoxon_path),
            "jackknife_csv": str(metrics_dir / "jackknife_auroc_by_fold.csv"),
            "cd_diagram": str(cd_fig),
        },
        "manuscript_guidance": (
            "Wilcoxon signed-rank on leave-one-participant-out jackknife AUROC vectors "
            "(paired across models); Holm correction for multiple baselines vs BiLSTM-AE. "
            "Critical Difference diagram: Friedman + Nemenyi on jackknife folds "
            "(Ismail Fawaz 2020 / Dempster 2019–2021 protocol)."
        ),
    }
    summary_path = metrics_dir / "statistical_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_markdown(metrics_dir / "statistical_benchmark_report.md", summary, wilcoxon_df, cd_summary)
    logger.info(
        "Statistical benchmark complete — {} models, {} jackknife folds → {}",
        len(model_names),
        len(pids),
        summary_path,
    )
    return summary


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    wilcoxon_df: pd.DataFrame,
    cd_summary: dict[str, Any],
) -> None:
    friedman = cd_summary["friedman"]
    lines = [
        "# Statistical benchmark — Wilcoxon + Critical Difference",
        "",
        "Leave-one-participant-out jackknife AUROC; paired Wilcoxon vs BiLSTM-AE with "
        "Holm correction; CD diagram (Friedman + Nemenyi).",
        "",
        f"- **Reference:** {DISPLAY_NAMES.get(summary['reference_model'], summary['reference_model'])}",
        f"- **Jackknife folds:** {summary['n_jackknife_folds']}",
        f"- **Friedman χ²:** {friedman.get('chi2', float('nan')):.4f} "
        f"(p = {friedman.get('p_value', float('nan')):.4g})",
        f"- **Critical difference (α=0.05):** {cd_summary['critical_difference']:.4f}",
        "",
        "## Wilcoxon vs BiLSTM-AE (Holm-corrected)",
        "",
        "| Baseline | Δ AUROC (ref−base) | p | p (Holm) | Significant |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in wilcoxon_df.itertuples(index=False):
        base = DISPLAY_NAMES.get(str(row.baseline), str(row.baseline))
        sig = "✓" if getattr(row, "significant_holm", False) else "—"
        lines.append(
            f"| {base} | {float(row.mean_delta_auroc):+.4f} | "
            f"{float(row.p_value):.4g} | {float(row.p_holm):.4g} | {sig} |"
        )
    lines.extend(["", "## Average ranks (CD diagram)", ""])
    avr = cd_summary["average_ranks"]
    for model in sorted(avr, key=avr.get):
        name = DISPLAY_NAMES.get(model, model)
        lines.append(f"- **{name}:** {avr[model]:.3f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
