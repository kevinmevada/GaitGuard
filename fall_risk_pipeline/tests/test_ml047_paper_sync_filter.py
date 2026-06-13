"""ML-047: paper sync filters metrics.csv by evaluation protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.paper_results_sync import (  # noqa: E402
    filter_deploy_schema_loso,
    filter_tabular_nested_loso,
)
from src.evaluation.primary_endpoint import (  # noqa: E402
    PROTOCOL_DEPLOY_GLOBAL,
    PROTOCOL_NESTED_RFECV,
)


def _sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "dl_tcn",
                "auc": 0.99,
                "evaluation_mode": "loso_dl",
                "feature_selection_protocol": "n/a",
            },
            {
                "model": "ensemble_soft_voting",
                "auc": 0.88,
                "evaluation_mode": "full_nested",
                "feature_selection_protocol": PROTOCOL_DEPLOY_GLOBAL,
            },
            {
                "model": "random_forest",
                "auc": 0.86,
                "evaluation_mode": "full_nested",
                "feature_selection_protocol": PROTOCOL_NESTED_RFECV,
            },
            {
                "model": "lightgbm",
                "auc": 0.84,
                "evaluation_mode": "full_nested",
                "feature_selection_protocol": PROTOCOL_NESTED_RFECV,
            },
        ]
    )


def test_filter_tabular_nested_loso_excludes_dl_and_deploy_protocol():
    df = _sample_metrics()
    tabular = filter_tabular_nested_loso(df)
    assert list(tabular["model"]) == ["random_forest", "lightgbm"]
    assert tabular.iloc[0]["model"] == "random_forest"


def test_filter_deploy_schema_loso_prefers_global_protocol():
    df = _sample_metrics()
    deploy = filter_deploy_schema_loso(df)
    assert len(deploy) == 1
    assert deploy.iloc[0]["model"] == "ensemble_soft_voting"


def test_primary_table_source_uses_filters():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "paper_results_sync.py").read_text(
        encoding="utf-8"
    )
    assert "filter_tabular_nested_loso" in source
    assert "PROTOCOL_NESTED_RFECV" in source
