#!/usr/bin/env python3
"""Recompute 2-method BiLSTM-AE ensemble scores from saved LOSO OOF.

Rank-averages Isolation Forest + One-Class SVM latent scores. Does not train
or load models; does not call main.py.

Usage:
    python scripts/recompute_ensemble_from_oof.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = PIPELINE_ROOT / "results" / "metrics" / "bilstm_ae_loso_oof_scores.csv"
DEFAULT_OUT = PIPELINE_ROOT / "results" / "metrics" / "bilstm_ae_loso_oof_scores_corrected.csv"

IF_COL = "isolation_forest_latent_score"
OCSVM_COL = "one_class_svm_latent_score"
ENSEMBLE_COL = "bilstm_ae_ensemble_score"
LABEL_COL = "eval_non_healthy"


def youden_threshold(y_true, y_score) -> float:
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    if len(pd.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int((tpr - fpr).argmax())
    thr = float(thresholds[idx])
    if not pd.notna(thr):
        return 0.5
    return max(0.0, min(1.0, thr))


def compute_metrics(y_true, scores) -> dict[str, float]:
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    thr = youden_threshold(y_true, scores)
    y_pred = (scores >= thr).astype(int)
    tn_fp = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    spec_den = tn_fp + fp
    sens_den = tp + fn
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(tp / sens_den) if sens_den else float("nan"),
        "specificity": float(tn_fp / spec_den) if spec_den else float("nan"),
        "youden_threshold": thr,
        "n_trials": int(len(y_true)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank-average IF + OCSVM latent OOF scores into a 2-method ensemble."
    )
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.oof)
    missing = {IF_COL, OCSVM_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise SystemExit(f"OOF file missing columns: {sorted(missing)}")

    if_rank = df[IF_COL].rank(pct=True)
    ocsvm_rank = df[OCSVM_COL].rank(pct=True)
    df[ENSEMBLE_COL] = 0.5 * (if_rank + ocsvm_rank)

    metrics = compute_metrics(df[LABEL_COL].to_numpy(), df[ENSEMBLE_COL].to_numpy())
    pred_col = "bilstm_ae_ensemble_pred"
    if pred_col in df.columns:
        df[pred_col] = (df[ENSEMBLE_COL] >= metrics["youden_threshold"]).astype(float)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote corrected OOF → {args.out}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
