"""ML-038: multiclass McNemar p-values labeled exploratory in metrics export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.classification_significance import (
    MCNEMAR_EXPLORATORY_SUFFIX,
    MCNEMAR_INTERPRETATION_MULTICLASS,
    format_mcnemar_pvalue_display,
    mcnemar_pvalue,
    pairwise_classification_significance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _multiclass_results() -> dict:
    y = np.array([0, 1, 2, 0, 1, 2])
    return {
        "m_a": {
            "y_true": y,
            "y_pred": np.array([0, 1, 2, 0, 1, 1]),
            "accuracy": float(np.mean(np.array([0, 1, 2, 0, 1, 1]) == y)),
            "participant_ids": np.arange(6),
            "label_mode": "multiclass",
        },
        "m_b": {
            "y_true": y,
            "y_pred": np.array([0, 1, 1, 0, 2, 2]),
            "accuracy": float(np.mean(np.array([0, 1, 1, 0, 2, 2]) == y)),
            "participant_ids": np.arange(6),
            "label_mode": "multiclass",
        },
    }


def test_pipeline_config_enables_exact_mcnemar():
    cfg = yaml.safe_load(
        (REPO_ROOT / "fall_risk_pipeline" / "configs" / "pipeline_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["models"]["evaluation"]["mcnemar_exact"] is True


def test_exact_mcnemar_differs_from_chi_squared_for_small_discordant_n():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    pred_a = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    pred_b = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    p_chi, _, _ = mcnemar_pvalue(y, pred_a, pred_b, exact=False)
    p_exact, _, _ = mcnemar_pvalue(y, pred_a, pred_b, exact=True)
    assert p_chi != p_exact


def test_format_mcnemar_pvalue_display_exploratory_suffix():
    assert format_mcnemar_pvalue_display(0.042, exploratory=True) == f"0.042{MCNEMAR_EXPLORATORY_SUFFIX}"
    assert format_mcnemar_pvalue_display(0.042, exploratory=False) == "0.042"
    assert format_mcnemar_pvalue_display("ref", exploratory=True) == "ref"


def test_multiclass_mcnemar_vs_reference_fmt_is_exploratory():
    _, vs_ref, _ = pairwise_classification_significance(
        _multiclass_results(),
        reference="m_a",
        apply_fdr=False,
        multiclass_mcnemar=True,
    )
    non_ref = vs_ref[vs_ref["model"] != "m_a"].iloc[0]
    assert non_ref["interpretation"] == MCNEMAR_INTERPRETATION_MULTICLASS
    assert str(non_ref["p_mcnemar_fmt"]).endswith(MCNEMAR_EXPLORATORY_SUFFIX)


def test_metrics_csv_row_would_carry_exploratory_fmt():
    """Simulate evaluator merge of mcnemar_vs_ref into metrics.csv."""
    _, vs_ref, _ = pairwise_classification_significance(
        _multiclass_results(),
        reference="m_a",
        apply_fdr=False,
        multiclass_mcnemar=True,
    )
    m_map = vs_ref.set_index("model").to_dict(orient="index")
    row = {
        "model": "m_b",
        "p_mcnemar_fmt": m_map["m_b"]["p_mcnemar_fmt"],
        "mcnemar_interpretation": m_map["m_b"]["interpretation"],
    }
    df = pd.DataFrame([row])
    assert df.loc[0, "p_mcnemar_fmt"].endswith(MCNEMAR_EXPLORATORY_SUFFIX)
    assert df.loc[0, "mcnemar_interpretation"] == MCNEMAR_INTERPRETATION_MULTICLASS
