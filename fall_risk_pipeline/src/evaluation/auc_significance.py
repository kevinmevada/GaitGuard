"""
Paired ROC AUC significance testing (DeLong + bootstrap Wilcoxon).

DeLong implementation adapted from Sun & Xu (2014), via:
https://github.com/yandexdataschool/roc_comparison (compare_auc_delong_xu.py)
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats
from loguru import logger
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014 fast algorithm)
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k
    t2 = np.empty(n, dtype=float)
    t2[j] = t + 1
    return t2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _delong_pvalue_from_log10(log10_p: np.ndarray) -> float:
    val = np.asarray(log10_p, dtype=float).reshape(-1)[0]
    return float(10 ** val)


def _compute_ground_truth_order(ground_truth: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.unique(ground_truth)
    if not np.array_equal(labels, [0, 1]) and not np.array_equal(labels, [0.0, 1.0]):
        raise ValueError("DeLong test requires binary labels {0, 1}")
    y = ground_truth.astype(int)
    order = (-y).argsort()
    return order, int(y.sum())


def delong_roc_pvalue(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> float:
    """Two-sided DeLong p-value for H0: AUC(prob_a) = AUC(prob_b)."""
    y = y_true.astype(int)
    order, label_1_count = _compute_ground_truth_order(y)
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, delongcov = _fast_delong(preds, label_1_count)
    l = np.array([[1.0, -1.0]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T))
    log10_p = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return _delong_pvalue_from_log10(log10_p)


def paired_bootstrap_auc_samples(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap AUC replicates (same resample indices for both models)."""
    rng = np.random.default_rng(random_state)
    y = y_true.astype(int)
    idx_all = np.arange(len(y))
    samples_a: list[float] = []
    samples_b: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples_a.append(float(roc_auc_score(yt, prob_a[idx])))
        samples_b.append(float(roc_auc_score(yt, prob_b[idx])))

    return np.asarray(samples_a), np.asarray(samples_b)


def bootstrap_auc_wilcoxon_pvalue(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> float:
    """Wilcoxon signed-rank on paired bootstrap AUC samples (pingouin)."""
    samples_a, samples_b = paired_bootstrap_auc_samples(
        y_true, prob_a, prob_b, n_bootstrap=n_bootstrap, random_state=random_state
    )
    if len(samples_a) < 10:
        return float("nan")
    result = pg.wilcoxon(samples_a, samples_b, alternative="two-sided")
    return float(result["p_val"].iloc[0])


def bootstrap_auc_mwu_pvalue(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Mann-Whitney U on paired bootstrap AUC samples via pingouin.mwu.

    Note: Wilcoxon signed-rank is the standard paired test; MWU is included
    as a reviewer-requested sensitivity check on bootstrap replicates.
    """
    samples_a, samples_b = paired_bootstrap_auc_samples(
        y_true, prob_a, prob_b, n_bootstrap=n_bootstrap, random_state=random_state
    )
    if len(samples_a) < 10:
        return float("nan")
    result = pg.mwu(samples_a, samples_b, alternative="two-sided")
    return float(result["p_val"].iloc[0])


def _format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def pairwise_auc_significance(
    results: dict[str, dict[str, Any]],
    *,
    reference: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise DeLong + bootstrap tests on nested-CV out-of-fold predictions.

    Returns
    -------
    pairwise_df : long-form pairwise p-values
    vs_reference_df : one row per model with p-values vs reference (best AUC if None)
    """
    if len(results) < 2:
        return pd.DataFrame(), pd.DataFrame()

    names = list(results.keys())
    y_ref = np.asarray(results[names[0]]["y_true"]).astype(int)
    if len(np.unique(y_ref)) > 2:
        logger.info(
            "Skipping pairwise DeLong/bootstrap AUC tests (multiclass labels; binary-only)."
        )
        return pd.DataFrame(), pd.DataFrame()

    probs: dict[str, np.ndarray] = {}

    for name in names:
        y_true = np.asarray(results[name]["y_true"]).astype(int)
        if y_true.shape != y_ref.shape or not np.array_equal(y_true, y_ref):
            raise ValueError(
                f"Out-of-fold labels for {name} do not align with {names[0]}; "
                "cannot run paired AUC tests."
            )
        probs[name] = np.asarray(results[name]["y_prob"], dtype=float)

    if reference is None:
        reference = max(names, key=lambda n: float(results[n]["auc"]))

    pairwise_rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(sorted(names), 2):
        p_delong = delong_roc_pvalue(y_ref, probs[model_a], probs[model_b])
        p_wilcoxon = bootstrap_auc_wilcoxon_pvalue(
            y_ref, probs[model_a], probs[model_b],
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        p_mwu = bootstrap_auc_mwu_pvalue(
            y_ref, probs[model_a], probs[model_b],
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        pairwise_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "auc_a": float(results[model_a]["auc"]),
            "auc_b": float(results[model_b]["auc"]),
            "delta_auc": float(results[model_a]["auc"] - results[model_b]["auc"]),
            "p_delong": p_delong,
            "p_bootstrap_wilcoxon": p_wilcoxon,
            "p_bootstrap_mwu": p_mwu,
            "p_delong_fmt": _format_pvalue(p_delong),
            "reference_pair": reference in (model_a, model_b),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)

    vs_rows: list[dict[str, Any]] = []
    for name in names:
        if name == reference:
            vs_rows.append({
                "model": name,
                "reference_model": reference,
                "p_delong_vs_reference": 1.0,
                "p_bootstrap_wilcoxon_vs_reference": 1.0,
                "p_bootstrap_mwu_vs_reference": 1.0,
                "p_delong_fmt": "ref",
            })
            continue
        p_delong = delong_roc_pvalue(y_ref, probs[name], probs[reference])
        p_wilcoxon = bootstrap_auc_wilcoxon_pvalue(
            y_ref, probs[name], probs[reference],
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        p_mwu = bootstrap_auc_mwu_pvalue(
            y_ref, probs[name], probs[reference],
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        vs_rows.append({
            "model": name,
            "reference_model": reference,
            "p_delong_vs_reference": p_delong,
            "p_bootstrap_wilcoxon_vs_reference": p_wilcoxon,
            "p_bootstrap_mwu_vs_reference": p_mwu,
            "p_delong_fmt": _format_pvalue(p_delong),
        })

    vs_reference_df = pd.DataFrame(vs_rows)
    logger.info(
        f"AUC pairwise tests complete (reference={reference}, "
        f"{len(pairwise_rows)} pairs)"
    )
    return pairwise_df, vs_reference_df
