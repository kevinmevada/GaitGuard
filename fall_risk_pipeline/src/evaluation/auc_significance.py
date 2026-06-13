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

from src.evaluation.multiclass_metrics import is_multiclass_metric_result

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
    if np.asarray(prob_a).ndim == 2:
        return paired_bootstrap_macro_auc_samples(
            y_true, prob_a, prob_b,
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
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


def paired_bootstrap_macro_auc_samples(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    labels: list[int] | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap macro-OVR AUC replicates for multiclass OOF predictions."""
    rng = np.random.default_rng(random_state)
    y = y_true.astype(int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    label_list = labels or sorted(set(y.tolist()))
    idx_all = np.arange(len(y))
    samples_a: list[float] = []
    samples_b: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt = y[idx]
        if len(np.unique(yt)) < len(label_list):
            continue
        try:
            samples_a.append(
                float(
                    roc_auc_score(
                        yt, proba_a[idx], multi_class="ovr", average="macro", labels=label_list
                    )
                )
            )
            samples_b.append(
                float(
                    roc_auc_score(
                        yt, proba_b[idx], multi_class="ovr", average="macro", labels=label_list
                    )
                )
            )
        except ValueError:
            continue

    return np.asarray(samples_a), np.asarray(samples_b)


def _result_probability_matrix(result: dict[str, Any]) -> np.ndarray:
    if is_multiclass_metric_result(result):
        return np.asarray(result["y_proba_full"], dtype=float)
    return np.asarray(result["y_prob"], dtype=float)


def _is_multiclass_results(results: dict[str, dict[str, Any]]) -> bool:
    first = next(iter(results.values()))
    y_true = np.asarray(first["y_true"]).astype(int)
    return len(np.unique(y_true)) > 2 or is_multiclass_metric_result(first)


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
    """Mann-Whitney U on paired bootstrap AUC samples (sensitivity check)."""
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


def benjamini_hochberg_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values for a vector of p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def apply_fdr_correction(
    df: pd.DataFrame,
    p_col: str,
    q_col: str,
) -> pd.DataFrame:
    """Add FDR q-value column in place (NaN p-values stay NaN)."""
    if df.empty or p_col not in df.columns:
        return df
    pvals = df[p_col].values.astype(float)
    mask = np.isfinite(pvals)
    qvals = np.full(len(pvals), np.nan, dtype=float)
    if mask.sum() > 0:
        qvals[mask] = benjamini_hochberg_qvalues(pvals[mask])
    df = df.copy()
    df[q_col] = qvals
    return df


def paired_bootstrap_auc_difference_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Subject-level paired bootstrap on macro/binary AUC difference (A − B).

    Primary exploratory test for LOSO OOF comparisons: resamples subjects
    with replacement and tests whether the bootstrap AUC delta distribution
    is centered at zero.
    """
    samples_a, samples_b = paired_bootstrap_auc_samples(
        y_true,
        prob_a,
        prob_b,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    n = len(samples_a)
    if n < 10:
        return {
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mean_delta_auc": float("nan"),
            "n_bootstrap": float(n),
        }
    deltas = samples_a - samples_b
    p = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    p = min(1.0, p)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "p_value": p,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mean_delta_auc": float(np.mean(deltas)),
        "n_bootstrap": float(n),
    }


def _align_paired_oof_probs(
    results: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """
    Align model OOF predictions to a common subject order (one row per participant).
    """
    names = list(results.keys())
    ref = results[names[0]]
    y_ref = np.asarray(ref["y_true"]).astype(int)
    pids = ref.get("participant_ids")
    if pids is not None and len(pids) == len(y_ref):
        order = np.argsort(np.asarray(pids, dtype=str))
        y_ref = y_ref[order]
        pid_order = np.asarray(pids, dtype=str)[order]
    else:
        order = np.arange(len(y_ref))
        pid_order = None

    probs: dict[str, np.ndarray] = {}
    for name in names:
        y_true = np.asarray(results[name]["y_true"]).astype(int)
        prob = _result_probability_matrix(results[name])
        if y_true.shape != np.asarray(ref["y_true"]).shape:
            raise ValueError(f"Out-of-fold labels for {name} do not align with {names[0]}")
        if pid_order is not None:
            pids_n = np.asarray(results[name].get("participant_ids", pids), dtype=str)
            if not np.array_equal(np.sort(pids_n), np.sort(pid_order)):
                raise ValueError(f"participant_ids for {name} do not match reference")
            idx = np.argsort(pids_n)
            y_true = y_true[idx]
            prob = prob[idx]
        if not np.array_equal(y_true, y_ref):
            raise ValueError(f"Out-of-fold labels for {name} do not align after sorting")
        probs[name] = prob

    return y_ref, probs, pid_order


def pairwise_auc_significance(
    results: dict[str, dict[str, Any]],
    *,
    reference: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    apply_fdr: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise AUC significance on LOSO out-of-fold predictions (exploratory).

    Primary p-value: paired **subject-level bootstrap** on AUC difference (A−B).
    Binary supplementary: DeLong on paired subjects. Wilcoxon/MWU on bootstrap
    AUC marginals are retained as sensitivity checks only.

    When ``apply_fdr`` is True, Benjamini-Hochberg q-values are added across
    all pairwise comparisons (ML-007).
    """
    if len(results) < 2:
        return pd.DataFrame(), pd.DataFrame()

    names = list(results.keys())
    y_ref, probs, _ = _align_paired_oof_probs(results)
    multiclass = _is_multiclass_results(results)

    if reference is None:
        reference = max(names, key=lambda n: float(results[n]["auc"]))

    pairwise_rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(sorted(names), 2):
        if multiclass:
            p_delong = float("nan")
            p_delong_fmt = "n/a (multiclass)"
        else:
            p_delong = delong_roc_pvalue(y_ref, probs[model_a], probs[model_b])
            p_delong_fmt = _format_pvalue(p_delong)

        delta_test = paired_bootstrap_auc_difference_test(
            y_ref,
            probs[model_a],
            probs[model_b],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
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
            "label_mode": "multiclass" if multiclass else "binary",
            "interpretation": "exploratory_loso_oof",
            "significance_method": "paired_subject_bootstrap_delta",
            "n_subjects": int(len(y_ref)),
            "auc_a": float(results[model_a]["auc"]),
            "auc_b": float(results[model_b]["auc"]),
            "delta_auc": float(results[model_a]["auc"] - results[model_b]["auc"]),
            "p_bootstrap_delta": delta_test["p_value"],
            "p_bootstrap_delta_fmt": _format_pvalue(delta_test["p_value"]),
            "bootstrap_delta_ci_low": delta_test["ci_low"],
            "bootstrap_delta_ci_high": delta_test["ci_high"],
            "bootstrap_mean_delta_auc": delta_test["mean_delta_auc"],
            "n_bootstrap_replicates": delta_test["n_bootstrap"],
            "p_delong": p_delong,
            "p_bootstrap_wilcoxon": p_wilcoxon,
            "p_bootstrap_mwu": p_mwu,
            "p_delong_fmt": p_delong_fmt,
            "reference_pair": reference in (model_a, model_b),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if apply_fdr and not pairwise_df.empty:
        pairwise_df = apply_fdr_correction(
            pairwise_df, "p_bootstrap_delta", "fdr_q_bootstrap_delta"
        )
        if "p_delong" in pairwise_df.columns:
            pairwise_df = apply_fdr_correction(
                pairwise_df, "p_delong", "fdr_q_delong"
            )

    vs_rows: list[dict[str, Any]] = []
    for name in names:
        if name == reference:
            vs_rows.append({
                "model": name,
                "reference_model": reference,
                "label_mode": "multiclass" if multiclass else "binary",
                "interpretation": "exploratory_loso_oof",
                "significance_method": "paired_subject_bootstrap_delta",
                "p_bootstrap_delta_vs_reference": 1.0,
                "p_bootstrap_delta_fmt": "ref",
                "p_delong_vs_reference": 1.0,
                "p_bootstrap_wilcoxon_vs_reference": 1.0,
                "p_bootstrap_mwu_vs_reference": 1.0,
                "p_delong_fmt": "ref",
            })
            continue
        if multiclass:
            p_delong = float("nan")
            p_delong_fmt = "n/a (multiclass)"
        else:
            p_delong = delong_roc_pvalue(y_ref, probs[name], probs[reference])
            p_delong_fmt = _format_pvalue(p_delong)

        delta_test = paired_bootstrap_auc_difference_test(
            y_ref,
            probs[name],
            probs[reference],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
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
            "label_mode": "multiclass" if multiclass else "binary",
            "interpretation": "exploratory_loso_oof",
            "significance_method": "paired_subject_bootstrap_delta",
            "n_subjects": int(len(y_ref)),
            "p_bootstrap_delta_vs_reference": delta_test["p_value"],
            "p_bootstrap_delta_fmt": _format_pvalue(delta_test["p_value"]),
            "bootstrap_delta_ci_low": delta_test["ci_low"],
            "bootstrap_delta_ci_high": delta_test["ci_high"],
            "p_delong_vs_reference": p_delong,
            "p_bootstrap_wilcoxon_vs_reference": p_wilcoxon,
            "p_bootstrap_mwu_vs_reference": p_mwu,
            "p_delong_fmt": p_delong_fmt,
        })

    vs_reference_df = pd.DataFrame(vs_rows)
    if apply_fdr and not vs_reference_df.empty:
        vs_reference_df = apply_fdr_correction(
            vs_reference_df,
            "p_bootstrap_delta_vs_reference",
            "fdr_q_bootstrap_delta_vs_reference",
        )

    mode = "paired subject bootstrap delta" if multiclass else "DeLong + paired bootstrap delta"
    logger.info(
        f"AUC pairwise tests complete ({mode}, reference={reference}, "
        f"{len(pairwise_rows)} pairs, exploratory LOSO OOF)"
    )
    return pairwise_df, vs_reference_df


def _oof_group_to_result(group: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score

    y_true = group["y_true"].values.astype(int)
    prob_cols = sorted(
        c for c in group.columns if c.startswith("y_prob_class_")
    )
    if prob_cols:
        y_proba = group[prob_cols].values.astype(float)
        auc = float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro",
                labels=sorted(np.unique(y_true)),
            )
        )
        return {
            "y_true": y_true,
            "y_proba_full": y_proba,
            "label_mode": "multiclass",
            "auc": auc,
            "accuracy": float(accuracy_score(y_true, np.argmax(y_proba, axis=1))),
        }
    y_prob = group["y_prob"].values.astype(float)
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
    }


def dl_vs_classical_auc_significance(
    metrics_dir,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    apply_fdr: bool = True,
) -> pd.DataFrame:
    """Bootstrap paired macro-OVR AUC: each DL model vs best classical ML model."""
    from pathlib import Path

    metrics_dir = Path(metrics_dir)
    cls_path = metrics_dir / "oof_predictions.parquet"
    dl_path = metrics_dir / "deep_learning_oof_predictions.parquet"
    if not cls_path.exists() or not dl_path.exists():
        return pd.DataFrame()

    cls_oof = pd.read_parquet(cls_path)
    dl_oof = pd.read_parquet(dl_path)
    classical: dict[str, dict[str, Any]] = {}
    for model, grp in cls_oof.groupby("model"):
        classical[str(model)] = _oof_group_to_result(grp)

    if not classical:
        return pd.DataFrame()

    best_cl = max(classical, key=lambda n: float(classical[n]["auc"]))
    ref = classical[best_cl]
    y_ref = np.asarray(ref["y_true"]).astype(int)
    ref_probs = _result_probability_matrix(ref)

    rows: list[dict[str, Any]] = []
    for dl_model, grp in dl_oof.groupby("model"):
        dl_res = _oof_group_to_result(grp)
        y_dl = np.asarray(dl_res["y_true"]).astype(int)
        if y_dl.shape != y_ref.shape or not np.array_equal(y_dl, y_ref):
            logger.warning(f"Skipping DL vs classical test for {dl_model}: label misalignment")
            continue
        dl_probs = _result_probability_matrix(dl_res)
        delta_test = paired_bootstrap_auc_difference_test(
            y_ref,
            dl_probs,
            ref_probs,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        p_wilcoxon = bootstrap_auc_wilcoxon_pvalue(
            y_ref, dl_probs, ref_probs,
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        p_mwu = bootstrap_auc_mwu_pvalue(
            y_ref, dl_probs, ref_probs,
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        rows.append({
            "dl_model": str(dl_model),
            "classical_reference": best_cl,
            "interpretation": "exploratory_loso_oof",
            "significance_method": "paired_subject_bootstrap_delta",
            "n_subjects": int(len(y_ref)),
            "auc_dl": float(dl_res["auc"]),
            "auc_classical": float(ref["auc"]),
            "delta_auc_dl_minus_classical": float(dl_res["auc"] - ref["auc"]),
            "p_bootstrap_delta": delta_test["p_value"],
            "p_bootstrap_delta_fmt": _format_pvalue(delta_test["p_value"]),
            "bootstrap_delta_ci_low": delta_test["ci_low"],
            "bootstrap_delta_ci_high": delta_test["ci_high"],
            "p_bootstrap_wilcoxon": p_wilcoxon,
            "p_bootstrap_mwu": p_mwu,
            "p_bootstrap_mwu_fmt": _format_pvalue(p_mwu),
            "label_mode": dl_res.get("label_mode", "binary"),
        })

    out = pd.DataFrame(rows)
    if apply_fdr and not out.empty:
        out = apply_fdr_correction(out, "p_bootstrap_delta", "fdr_q_bootstrap_delta")
    if not out.empty:
        out.to_csv(metrics_dir / "dl_vs_classical_pairwise_pvalues.csv", index=False)
        logger.info(
            f"DL vs classical AUC tests saved ({len(out)} DL models vs {best_cl})"
        )
    return out
