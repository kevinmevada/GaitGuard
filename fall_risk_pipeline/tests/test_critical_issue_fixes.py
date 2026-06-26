"""Tests for critical audit fixes (DEP-001, PUB-001)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_dockerfile_api_serves_static_ui():
    dockerfile = (REPO_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "COPY api /app/api" in dockerfile
    assert "sync_front_end" not in dockerfile


def test_pipeline_requirements_include_api_and_download_deps():
    req = (REPO_ROOT / "fall_risk_pipeline" / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" in req
    assert "joblib" in req
    assert "fastapi" in req
    assert "requests" in req
    assert "huggingface_hub" in req


def test_regenerate_paper_results_script_exists():
    script = REPO_ROOT / "scripts" / "regenerate_paper_results.py"
    assert script.is_file()


def test_paper_results_stub_has_no_stale_5fold_claims():
    results = (REPO_ROOT / "docs" / "paper" / "results.md").read_text(encoding="utf-8").lower()
    assert "5-fold" not in results
    assert "stratifiedgroupkfold" not in results or "no longer match" in results
    assert "0.9336" not in results
    assert "0.9164" not in results


def test_abstract_has_no_hardcoded_stale_auc():
    abstract = (REPO_ROOT / "docs" / "paper" / "abstract.md").read_text(encoding="utf-8")
    assert "0.91" not in abstract
    assert "0.9336" not in abstract
    assert "regenerate_paper_results" in abstract


def test_introduction_has_no_hardcoded_stale_auc():
    intro = (REPO_ROOT / "docs" / "paper" / "introduction.md").read_text(encoding="utf-8")
    assert "0.934" not in intro
    assert "0.9336" not in intro
    assert "docs/paper/results.md" in intro


def test_readmes_have_no_hardcoded_stale_auc():
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pipeline_readme = (PIPELINE_ROOT / "README.md").read_text(encoding="utf-8")
    for text in (root_readme, pipeline_readme):
        assert "0.9336" not in text
        assert "0.9273" not in text
        assert "docs/paper/results.md" in text
    assert "P/N ≈ 3.25" not in root_readme


def test_discussion_has_no_stale_run_specific_claims():
    discussion = (REPO_ROOT / "docs" / "paper" / "discussion.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "+2.33%" not in discussion
    assert "lb_range_ap_std" not in discussion
    assert "tcn performing strongest" not in discussion
    assert "regenerate" in discussion or "regenerated" in discussion


def test_paper_results_sync_module_importable():
    import sys

    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.paper_results_sync import (  # noqa: WPS433
        MISSING_ARTIFACTS_STUB,
        build_paper_results_md,
        sync_paper_results,
    )

    assert "LOSO" in MISSING_ARTIFACTS_STUB
    assert callable(build_paper_results_md)
    assert callable(sync_paper_results)


def test_sync_paper_results_writes_stub_without_metrics(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation import paper_results_sync as prs

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    config = {"paths": {"metrics": str(metrics_dir)}}

    out = tmp_path / "paper" / "results.md"
    monkeypatch.setattr(prs, "PAPER_RESULTS_PATH", out)

    prs.sync_paper_results(config)
    text = out.read_text(encoding="utf-8")
    assert "metrics.csv" in text
    assert "5-fold" not in text.lower()


def test_sync_paper_results_from_metrics_csv(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation import paper_results_sync as prs

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.csv").write_text(
        "model,auc,auc_ci_low,auc_ci_high,f1,accuracy,evaluation_mode,feature_selection_protocol\n"
        "random_forest,0.8600,0.8300,0.8800,0.7200,0.7600,full_nested,nested_rfecv_per_loso_fold\n"
        "ensemble_soft_voting,0.8500,0.8200,0.8700,0.7100,0.7500,full_nested,global_selected_features_json\n"
        "dl_tcn,0.9900,0.9800,0.9950,0.9000,0.9100,loso_dl,n/a\n",
        encoding="utf-8",
    )
    config = {"paths": {"metrics": str(metrics_dir)}}

    out = tmp_path / "paper" / "results.md"
    abstract = tmp_path / "paper" / "abstract.md"
    abstract.parent.mkdir(parents=True, exist_ok=True)
    abstract.write_text("# Abstract\n\n## Metrics fill-in\n\nold\n", encoding="utf-8")
    monkeypatch.setattr(prs, "PAPER_RESULTS_PATH", out)
    monkeypatch.setattr(prs, "PAPER_ABSTRACT_PATH", abstract)

    prs.sync_paper_results(config)
    results = out.read_text(encoding="utf-8")
    assert "0.8600" in results
    assert "random_forest" in results
    assert "dl_tcn" not in results
    assert "LOSO" in results
    assert "5-fold" not in results.lower()

    abs_text = abstract.read_text(encoding="utf-8")
    assert "0.8600" in abs_text
    assert "random_forest" in abs_text
