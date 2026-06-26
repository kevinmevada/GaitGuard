"""Regression checks for medium-severity audit fixes (ML-023–ML-029, REP, COD, STR, SEC)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_trainer_has_nested_ensemble_cv():
    source = (PIPELINE_ROOT / "src" / "models" / "trainer.py").read_text(encoding="utf-8")
    assert "_nested_ensemble_cv" in source
    assert "nested_rfecv_ensemble_cv" in source


def test_evaluator_tune_top_k_uses_loso_ranking():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(encoding="utf-8")
    assert "return self._select_top_base_models(tuned_results, top_k)" in source


def test_nested_in_ablation_config_enabled():
    cfg = yaml.safe_load(
        (PIPELINE_ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
    )
    fs = cfg["feature_selection"]
    assert fs.get("nested_in_ablation") is True
    assert int(fs.get("max_required_features", 99)) <= 4


def test_feature_matrix_intersect_helper():
    sys_path_added = str(PIPELINE_ROOT)
    import sys

    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)
    from src.features.feature_matrix import intersect_nested_rfecv_columns

    assert callable(intersect_nested_rfecv_columns)


def test_ablations_import_nested_intersect():
    fa = (PIPELINE_ROOT / "src" / "evaluation" / "feature_ablation.py").read_text(
        encoding="utf-8"
    )
    sa = (PIPELINE_ROOT / "src" / "evaluation" / "sensor_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "intersect_nested_rfecv_columns" in fa
    assert "intersect_nested_rfecv_columns" in sa


def test_multiclass_stacking_uses_group_stacking():
    source = (PIPELINE_ROOT / "src" / "models" / "ensemble_builder.py").read_text(
        encoding="utf-8"
    )
    assert "_mean_base_proba" not in source
    assert "GroupStackingEnsemble" in source
    assert "ML-026" in source or "binary and multiclass" in source


def test_evaluator_runs_mcnemar_for_multiclass():
    source = (PIPELINE_ROOT / "src" / "evaluation" / "evaluator.py").read_text(encoding="utf-8")
    assert "mcnemar_pairwise_df, mcnemar_vs_ref_df, fold_disc_df = self._run_mcnemar_tests" in source
    assert "Skipping DeLong/McNemar (not yet implemented for multiclass)" not in source


def test_feature_selection_report_uses_computed_p_n_ratio():
    source = (PIPELINE_ROOT / "src" / "features" / "feature_selector.py").read_text(
        encoding="utf-8"
    )
    assert "3.25" not in source
    assert "p_n_before = n_features_before / max(n_participants, 1)" in source
    assert "payload['p_n_ratio_before']" in source
    assert "payload['n_participants']" in source

    from src.features.feature_selector import FeatureSelector

    selector = FeatureSelector(
        {
            "paths": {"features": "data/features", "metrics": "results/metrics"},
            "models": {
                "tuning": {"cv_folds": 5},
                "evaluation": {"random_state": 42},
            },
            "feature_selection": {"max_features": 20},
        }
    )
    payload = {
        "n_participants": 260,
        "n_features_before": 464,
        "n_features_after": 20,
        "p_n_ratio_before": 1.785,
        "p_n_ratio_after": 0.077,
        "max_features": 20,
        "primary_method": "rfecv_capped",
        "rfecv_capped_to_max_features": True,
        "forced_required_features": [],
        "dropped_required_features": [],
        "rfecv_detail": {"n_features_cv_optimal": 45},
        "rfecv_features": ["f1"],
        "features": ["f1"],
        "comparison": [],
    }
    selector._write_report_md(payload)
    report = (Path("results/metrics") / "feature_selection_report.md").read_text(
        encoding="utf-8"
    )
    assert "P/N ratio before (p/N): **1.78**" in report
    assert "P/N ratio after (p/N): **0.08**" in report
    assert "P/N \u2248 1.78" in report
    assert "reduce P/N to \u2248 0.08" in report
    assert "3.25" not in report
    assert "0.56" not in report


def test_feature_selection_comparison_flags():
    source = (PIPELINE_ROOT / "src" / "features" / "feature_selector.py").read_text(
        encoding="utf-8"
    )
    assert '"nested_selection": False' in source
    assert '"global_rfecv_mask"' in source


def test_requirements_dev_split():
    main = (PIPELINE_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    dev = (PIPELINE_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()
    assert "pytest>=" in dev
    assert "pytest>=" not in main
    assert "pytest-cov" in dev
    assert "pytest-cov" not in main
    assert "torchvision" not in main
    for unused in ("plotly", "tabulate", "jinja2", "optuna-dashboard"):
        assert unused not in main
        assert unused not in dev


def test_anomaly_detector_uses_train_fold_healthy_fit():
    source = (PIPELINE_ROOT / "src" / "models" / "anomaly_detector.py").read_text(
        encoding="utf-8"
    )
    assert "healthy_train_fit_mask" in source
    assert "reconstruction_threshold_train_only" in source
    assert "fit_mask" in source


def test_lockfile_generator_exists():
    assert (REPO_ROOT / "scripts" / "generate_requirements_lock.py").is_file()


def test_ci_pythonhashseed_matches_pipeline():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'PYTHONHASHSEED: "42"' in ci


def test_ci_python_matrix_includes_310_311_312():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for version in ("3.10", "3.11", "3.12"):
        assert version in ci


def test_api_static_mount_and_async_predict():
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    assert "StaticFiles" in source
    assert 'app.mount("/static"' in source
    assert "asyncio.to_thread" in source
    assert "_run_prediction_pipeline" in source
    assert '@app.get("/app")' in source
    assert '@app.post("/app/predict")' in source
    assert "_is_same_origin_ui_request" in source


def test_docker_non_root_and_slim_inference_deps():
    docker = (REPO_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "USER gaitguard" in docker
    assert "requirements-inference.txt" in docker
    assert "PYTHONHASHSEED=42" in docker
    assert "fall_risk_requirements.txt" not in docker


def test_training_dockerfile_hash_seed_matches_pipeline():
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "PYTHONHASHSEED=42" in docker
    assert "PYTHONHASHSEED=0" not in docker


def test_methods_documents_multiclass_mcnemar():
    methods = (REPO_ROOT / "docs" / "paper" / "methods.md").read_text(encoding="utf-8")
    assert "McNemar tests use argmax" in methods
    assert "nested_in_ablation" in methods or "per-fold nested RFECV" in methods


def test_paper_stale_metrics_disclaimed():
    discussion = (REPO_ROOT / "docs" / "paper" / "discussion.md").read_text(encoding="utf-8")
    assert "PUB-002" in discussion or "RES-002" in discussion
    assert "0.916" not in discussion


def test_mcnemar_multiclass_predictions_from_y_pred():
    import sys

    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.classification_significance import predictions_from_result

    preds = predictions_from_result({"y_pred": [0, 1, 2], "y_true": [0, 1, 2]})
    assert preds.tolist() == [0, 1, 2]


def test_trainer_nested_ensemble_cv_ast():
    tree = ast.parse(
        (PIPELINE_ROOT / "src" / "models" / "trainer.py").read_text(encoding="utf-8")
    )
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_nested_ensemble_cv" in names
