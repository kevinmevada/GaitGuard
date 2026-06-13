"""
Paired classification significance (McNemar) on LOSO out-of-fold predictions.

Each participant contributes exactly one held-out prediction per model (LOSO).
Discordant counts are aggregated across all folds into a single 2×2 table, then
``statsmodels.stats.contingency_tables.mcnemar`` is applied — equivalent to
summing per-fold confusion-matrix disagreements when folds are non-overlapping.

For multiclass tasks, McNemar is applied to argmax class predictions using
correct-vs-wrong discordant pairs only. This is an exploratory accuracy comparison,
not a Stuart–Maxwell test for full multiclass contingency tables.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.stats.contingency_tables import mcnemar

from src.evaluation.auc_significance import _format_pvalue, apply_fdr_correction

MCNEMAR_EXPLORATORY_SUFFIX = " (expl.)"
MCNEMAR_INTERPRETATION_BINARY = "loso_oof_binary"
MCNEMAR_INTERPRETATION_MULTICLASS = "multiclass_argmax_exploratory"


def format_mcnemar_pvalue_display(p: float | str, *, exploratory: bool = False) -> str:
    """Format McNemar p-value for export; multiclass argmax tests are labeled exploratory."""
    if isinstance(p, str):
        base = p
    else:
        base = _format_pvalue(p)
    if exploratory and base not in ("ref", "—", ""):
        return f"{base}{MCNEMAR_EXPLORATORY_SUFFIX}"
    return base


def predictions_from_result(res: dict[str, Any]) -> np.ndarray:
    """
    Class predictions from stored OOF outputs.

    Binary: thresholded ``y_prob`` (or ``y_pred`` when present).
    Multiclass: argmax ``y_pred`` — McNemar on these is exploratory (ML-038).
    """
    if "y_pred" in res:
        return np.asarray(res["y_pred"], dtype=int)
    y_prob = np.asarray(res["y_prob"], dtype=float)
    threshold = float(res.get("decision_threshold", 0.5))
    return (y_prob >= threshold).astype(int)


def mcnemar_discordant_counts(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Build McNemar discordant counts from paired predictions.

    Returns (n_01, n_10, both_correct, both_wrong) where
    n_01: model A wrong & model B correct; n_10: model A correct & model B wrong.
    """
    y = y_true.astype(int)
    pa = pred_a.astype(int)
    pb = pred_b.astype(int)
    correct_a = pa == y
    correct_b = pb == y
    n_01 = int(np.sum(~correct_a & correct_b))
    n_10 = int(np.sum(correct_a & ~correct_b))
    both_correct = int(np.sum(correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    return n_01, n_10, both_correct, both_wrong


def mcnemar_pvalue(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    exact: bool = False,
    correction: bool = True,
) -> tuple[float, int, int]:
    """
    Two-sided McNemar test for paired classifier agreement.

    Uses the standard discordant-only table [[0, n_01], [n_10, 0]].
    """
    n_01, n_10, _, _ = mcnemar_discordant_counts(y_true, pred_a, pred_b)
    if n_01 + n_10 == 0:
        return 1.0, n_01, n_10
    table = [[0, n_01], [n_10, 0]]
    result = mcnemar(table, exact=exact, correction=correction)
    return float(result.pvalue), n_01, n_10


def per_fold_discordant_rows(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    participant_ids: np.ndarray | None,
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    """Per-LOSO-fold discordant contributions (one held-out subject per fold)."""
    if participant_ids is None:
        return []

    rows: list[dict[str, Any]] = []
    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        if int(mask.sum()) != 1:
            continue
        n_01, n_10, bc, bw = mcnemar_discordant_counts(
            y_true[mask], pred_a[mask], pred_b[mask]
        )
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "participant_id": str(pid),
            "n_01": n_01,
            "n_10": n_10,
            "both_correct": bc,
            "both_wrong": bw,
        })
    return rows


def pairwise_classification_significance(
    results: dict[str, dict[str, Any]],
    *,
    reference: str | None = None,
    exact_mcnemar: bool = False,
    apply_fdr: bool = True,
    multiclass_mcnemar: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pairwise McNemar tests on pooled LOSO out-of-fold classifications.

    When ``multiclass_mcnemar`` is True, p-values are labeled exploratory in
    ``p_mcnemar_fmt`` (correct-vs-wrong argmax pairs; not Stuart–Maxwell).
    """
    if len(results) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    names = list(results.keys())
    y_ref = np.asarray(results[names[0]]["y_true"]).astype(int)
    preds: dict[str, np.ndarray] = {}
    pids = np.asarray(results[names[0]].get("participant_ids", []), dtype=object)

    for name in names:
        y_true = np.asarray(results[name]["y_true"]).astype(int)
        if y_true.shape != y_ref.shape or not np.array_equal(y_true, y_ref):
            raise ValueError(f"OOF labels for {name} do not align with {names[0]}")
        preds[name] = predictions_from_result(results[name])

    if reference is None:
        reference = max(names, key=lambda n: float(results[n]["accuracy"]))

    interpretation = (
        MCNEMAR_INTERPRETATION_MULTICLASS
        if multiclass_mcnemar
        else MCNEMAR_INTERPRETATION_BINARY
    )

    pairwise_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for model_a, model_b in combinations(sorted(names), 2):
        p_val, n_01, n_10 = mcnemar_pvalue(
            y_ref, preds[model_a], preds[model_b], exact=exact_mcnemar
        )
        _, _, bc, bw = mcnemar_discordant_counts(y_ref, preds[model_a], preds[model_b])
        acc_a = float(np.mean(preds[model_a] == y_ref))
        acc_b = float(np.mean(preds[model_b] == y_ref))

        pairwise_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "interpretation": interpretation,
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "delta_accuracy": acc_a - acc_b,
            "n_01_b_correct_a_wrong": n_01,
            "n_10_a_correct_b_wrong": n_10,
            "both_correct": bc,
            "both_wrong": bw,
            "p_mcnemar": p_val,
            "p_mcnemar_fmt": format_mcnemar_pvalue_display(
                p_val, exploratory=multiclass_mcnemar
            ),
            "reference_pair": reference in (model_a, model_b),
        })
        if len(pids) == len(y_ref):
            fold_rows.extend(
                per_fold_discordant_rows(
                    y_ref, preds[model_a], preds[model_b], pids, model_a, model_b
                )
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if apply_fdr and not pairwise_df.empty:
        pairwise_df = apply_fdr_correction(pairwise_df, "p_mcnemar", "fdr_q_mcnemar")

    vs_rows: list[dict[str, Any]] = []
    for name in names:
        if name == reference:
            vs_rows.append({
                "model": name,
                "reference_model": reference,
                "p_mcnemar_vs_reference": 1.0,
                "p_mcnemar_fmt": "ref",
                "n_01_vs_reference": 0,
                "n_10_vs_reference": 0,
            })
            continue
        p_val, n_01, n_10 = mcnemar_pvalue(
            y_ref, preds[name], preds[reference], exact=exact_mcnemar
        )
        vs_rows.append({
            "model": name,
            "reference_model": reference,
            "interpretation": interpretation,
            "p_mcnemar_vs_reference": p_val,
            "p_mcnemar_fmt": format_mcnemar_pvalue_display(
                p_val, exploratory=multiclass_mcnemar
            ),
            "n_01_vs_reference": n_01,
            "n_10_vs_reference": n_10,
        })

    vs_reference_df = pd.DataFrame(vs_rows)
    if apply_fdr and not vs_reference_df.empty:
        vs_reference_df = apply_fdr_correction(
            vs_reference_df, "p_mcnemar_vs_reference", "fdr_q_mcnemar_vs_reference"
        )
    fold_discordant_df = pd.DataFrame(fold_rows)

    logger.info(
        f"McNemar tests complete (reference={reference}, {len(pairwise_rows)} pairs)"
    )
    return pairwise_df, vs_reference_df, fold_discordant_df
