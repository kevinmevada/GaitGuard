"""Multiclass paired bootstrap AUC significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.auc_significance import (
    _oof_group_to_result,
    dl_vs_classical_auc_significance,
    paired_bootstrap_macro_auc_samples,
    pairwise_auc_significance,
)


def _multiclass_results(n: int = 60, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    proba_a = np.zeros((n, 3))
    proba_b = np.zeros((n, 3))
    for i, label in enumerate(y):
        proba_a[i, label] = 0.7
        proba_a[i] += rng.random(3) * 0.1
        proba_a[i] /= proba_a[i].sum()
        proba_b[i, label] = 0.55
        proba_b[i] += rng.random(3) * 0.15
        proba_b[i] /= proba_b[i].sum()

    def pack(name: str, proba: np.ndarray) -> dict:
        auc = float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
        )
        return {"y_true": y, "y_proba_full": proba, "label_mode": "multiclass", "auc": auc}

    return {"model_a": pack("model_a", proba_a), "model_b": pack("model_b", proba_b)}


def test_paired_bootstrap_macro_auc_samples_returns_finite_replicates():
    results = _multiclass_results()
    y = results["model_a"]["y_true"]
    sa, sb = paired_bootstrap_macro_auc_samples(
        y,
        results["model_a"]["y_proba_full"],
        results["model_b"]["y_proba_full"],
        n_bootstrap=200,
        random_state=42,
    )
    assert len(sa) >= 10
    assert len(sa) == len(sb)
    assert np.all(np.isfinite(sa))
    assert np.all(np.isfinite(sb))


def test_pairwise_auc_significance_multiclass_omits_delong():
    results = _multiclass_results(n=120, seed=7)
    pairwise_df, vs_ref_df = pairwise_auc_significance(
        results, n_bootstrap=400, random_state=42
    )
    assert not pairwise_df.empty
    assert pairwise_df.iloc[0]["label_mode"] == "multiclass"
    assert np.isnan(pairwise_df.iloc[0]["p_delong"])
    assert pairwise_df.iloc[0]["p_delong_fmt"] == "n/a (multiclass)"
    assert np.isfinite(pairwise_df.iloc[0]["p_bootstrap_mwu"])
    assert not vs_ref_df.empty
    assert vs_ref_df.iloc[0]["label_mode"] == "multiclass"


def test_oof_group_to_result_multiclass_columns():
    rng = np.random.default_rng(1)
    n = 30
    y = rng.integers(0, 3, size=n)
    rows = []
    for i in range(n):
        row = {"y_true": int(y[i]), "y_pred": int(y[i])}
        probs = rng.random(3)
        probs /= probs.sum()
        for c in range(3):
            row[f"y_prob_class_{c}"] = float(probs[c])
        rows.append(row)
    grp = pd.DataFrame(rows)
    res = _oof_group_to_result(grp)
    assert res["label_mode"] == "multiclass"
    assert res["y_proba_full"].shape == (n, 3)
    assert np.isfinite(res["auc"])


def test_dl_vs_classical_auc_significance(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    n = 40
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=n)
    pid = np.arange(n)

    def write_oof(path, model: str, shift: float):
        rows = []
        for i in range(n):
            row = {
                "model": model,
                "participant_id": int(pid[i]),
                "y_true": int(y[i]),
                "y_pred": int(y[i]),
            }
            probs = rng.random(3) + shift
            probs[int(y[i])] += 0.5
            probs /= probs.sum()
            for c in range(3):
                row[f"y_prob_class_{c}"] = float(probs[c])
            rows.append(row)
        pd.DataFrame(rows).to_parquet(path, index=False)

    write_oof(metrics_dir / "oof_predictions.parquet", "xgboost", 0.0)
    write_oof(metrics_dir / "deep_learning_oof_predictions.parquet", "dl_cnn", 0.2)

    out = dl_vs_classical_auc_significance(
        metrics_dir, n_bootstrap=150, random_state=42
    )
    assert not out.empty
    assert "p_bootstrap_mwu" in out.columns
    assert np.isfinite(out.iloc[0]["p_bootstrap_mwu"])
    assert (metrics_dir / "dl_vs_classical_pairwise_pvalues.csv").exists()
