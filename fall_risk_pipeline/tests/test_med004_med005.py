"""MED-004 / MED-005: EDA t-SNE caption and required-feature documentation."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_tsne_caption_constant_and_save_hook():
    from src.visualization.eda import TSNE_CAPTION

    assert "visualization only" in TSNE_CAPTION.lower()
    assert "per-fold normalization" in TSNE_CAPTION.lower()
    source = inspect.getsource(
        __import__("src.visualization.eda", fromlist=["EDAAnalyzer"]).EDAAnalyzer._plot_tsne
    )
    assert "caption=TSNE_CAPTION" in source


def test_eda_save_writes_caption_sidecar(tmp_path):
    import matplotlib.pyplot as plt

    from src.visualization.eda import EDAAnalyzer, TSNE_CAPTION

    config = {
        "paths": {
            "processed_data": str(tmp_path),
            "features": str(tmp_path),
            "figures_eda": str(tmp_path / "eda"),
        },
        "reporting": {"figure_format": "png", "figure_dpi": 80},
        "dataset": {"sampling_rate": 100},
    }
    eda = EDAAnalyzer(config)
    fig, _ = plt.subplots()
    eda._save(fig, "tsne_test", caption=TSNE_CAPTION)
    assert (tmp_path / "eda" / "tsne_test_caption.txt").exists()
    text = (tmp_path / "eda" / "tsne_test_caption.txt").read_text(encoding="utf-8")
    assert "StandardScaler" not in text  # caption describes behavior, not class name
    assert "per-fold normalization" in text


def test_plot_tsne_with_synthetic_parquet(tmp_path):
    from src.visualization.eda import EDAAnalyzer

    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    df = pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(30)],
            "cohort": ["Healthy"] * 10 + ["PD"] * 10 + ["HipOA"] * 10,
            "risk_label": [0] * 10 + [2] * 10 + [1] * 10,
            "feat_a": np.random.default_rng(0).normal(size=30),
            "feat_b": np.random.default_rng(1).normal(size=30),
        }
    )
    path = feat_dir / "patient_features.parquet"
    df.to_parquet(path, index=False)

    out_dir = tmp_path / "eda"
    config = {
        "paths": {
            "processed_data": str(tmp_path),
            "features": str(feat_dir),
            "figures_eda": str(out_dir),
        },
        "reporting": {"figure_format": "png", "figure_dpi": 80},
        "dataset": {"sampling_rate": 100},
    }
    eda = EDAAnalyzer(config)
    eda._plot_tsne(path, pd.DataFrame())
    assert (out_dir / "tsne.png").exists()
    assert (out_dir / "tsne_caption.txt").exists()


def test_config_documents_required_feature_sensitivity():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    fs = cfg["feature_selection"]
    assert fs["required_feature_substrings"] == ["sampen", "dfa"]
    assert int(fs["max_required_features"]) <= 4


def test_methods_document_required_nonlinear_retention():
    text = (REPO_ROOT / "docs" / "paper" / "methods.md").read_text(encoding="utf-8")
    assert "required_feature_shap_audit.csv" in text
    assert "required_feature_substrings: []" in text or "required_feature_substrings: []" in text.replace("'", '"')
    assert "MED-005" in text or "regardless of RFECV rank" in text


def test_limitations_document_tsne_scaling():
    text = (REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    assert "MED-004" in text
    assert "tsne_caption.txt" in text
