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


def pairwise_auc_significance(
    results: dict[str, dict[str, Any]],
    *,
    reference: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise DeLong + bootstrap tests on nested-CV out-of-fold predictions.

    Multiclass: DeLong is omitted (binary-only); paired bootstrap uses macro-OVR AUC.
    """
    if len(results) < 2:
        return pd.DataFrame(), pd.DataFrame()

    names = list(results.keys())
    y_ref = np.asarray(results[names[0]]["y_true"]).astype(int)
    multiclass = _is_multiclass_results(results)

    probs: dict[str, np.ndarray] = {}
    for name in names:
        y_true = np.asarray(results[name]["y_true"]).astype(int)
        if y_true.shape != y_ref.shape or not np.array_equal(y_true, y_ref):
            raise ValueError(
                f"Out-of-fold labels for {name} do not align with {names[0]}; "
                "cannot run paired AUC tests."
            )
        probs[name] = _result_probability_matrix(results[name])

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
            "auc_a": float(results[model_a]["auc"]),
            "auc_b": float(results[model_b]["auc"]),
            "delta_auc": float(results[model_a]["auc"] - results[model_b]["auc"]),
            "p_delong": p_delong,
            "p_bootstrap_wilcoxon": p_wilcoxon,
            "p_bootstrap_mwu": p_mwu,
            "p_delong_fmt": p_delong_fmt,
            "reference_pair": reference in (model_a, model_b),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)

    vs_rows: list[dict[str, Any]] = []
    for name in names:
        if name == reference:
            vs_rows.append({
                "model": name,
                "reference_model": reference,
                "label_mode": "multiclass" if multiclass else "binary",
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
            "p_delong_vs_reference": p_delong,
            "p_bootstrap_wilcoxon_vs_reference": p_wilcoxon,
            "p_bootstrap_mwu_vs_reference": p_mwu,
            "p_delong_fmt": p_delong_fmt,
        })

    vs_reference_df = pd.DataFrame(vs_rows)
    mode = "macro-OVR bootstrap" if multiclass else "DeLong+bootstrap"
    logger.info(
        f"AUC pairwise tests complete ({mode}, reference={reference}, "
        f"{len(pairwise_rows)} pairs)"
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
            "auc_dl": float(dl_res["auc"]),
            "auc_classical": float(ref["auc"]),
            "delta_auc_dl_minus_classical": float(dl_res["auc"] - ref["auc"]),
            "p_bootstrap_wilcoxon": p_wilcoxon,
            "p_bootstrap_mwu": p_mwu,
            "p_bootstrap_mwu_fmt": _format_pvalue(p_mwu),
            "label_mode": dl_res.get("label_mode", "binary"),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(metrics_dir / "dl_vs_classical_pairwise_pvalues.csv", index=False)
        logger.info(
            f"DL vs classical AUC tests saved ({len(out)} DL models vs {best_cl})"
        )
    return out
