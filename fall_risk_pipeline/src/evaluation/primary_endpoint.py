"""
Primary manuscript / API endpoint registry (ML-032).

LOSO nested RFECV (research) and global selected_features.json (deploy/API) use
different feature schemas by design. This module pre-registers both endpoints and
exports ``deploy_loso_gap.csv`` so Table 2 and API scores stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.ensemble_builder import primary_ensemble_checkpoint_name

ENDPOINT_DEPLOY_ENSEMBLE = "deploy_ensemble"
ENDPOINT_BEST_LOSO_NESTED = "best_loso_nested"
ENDPOINT_ANOMALY_ENSEMBLE = "anomaly_ensemble"

PROTOCOL_NESTED_RFECV = "nested_rfecv_per_loso_fold"
PROTOCOL_DEPLOY_GLOBAL = "global_selected_features_json"
PROTOCOL_ANOMALY_LOSO = "anomaly_loso_healthy_reference"

_ALL_ENDPOINTS = (
    ENDPOINT_DEPLOY_ENSEMBLE,
    ENDPOINT_BEST_LOSO_NESTED,
    ENDPOINT_ANOMALY_ENSEMBLE,
)


def resolve_primary_endpoint(config: dict) -> str:
    ep = config.get("models", {}).get("evaluation", {}).get(
        "primary_endpoint", ENDPOINT_DEPLOY_ENSEMBLE
    )
    if ep not in _ALL_ENDPOINTS:
        return ENDPOINT_DEPLOY_ENSEMBLE
    return ep


def report_deploy_loso_gap_enabled(config: dict) -> bool:
    return bool(
        config.get("models", {}).get("evaluation", {}).get("report_deploy_loso_gap", True)
    )


def deploy_ensemble_model_name(config: dict) -> str:
    return primary_ensemble_checkpoint_name(config)


def best_nested_model_name(nested_results: dict[str, dict]) -> str:
    return max(nested_results, key=lambda n: nested_results[n]["auc"])


def build_deploy_loso_gap_rows(
    nested_results: dict[str, dict],
    deploy_results: dict[str, dict],
    *,
    nested_feature_count_median: int | None = None,
    deploy_feature_count: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in nested_results:
        deploy = deploy_results.get(name)
        if deploy is None:
            continue
        nested = nested_results[name]
        nested_auc = float(nested["auc"])
        deploy_auc = float(deploy["auc"])
        rows.append({
            "model": name,
            "loso_auc_nested_rfecv": nested_auc,
            "loso_auc_deploy_schema": deploy_auc,
            "delta_auc_deploy_minus_nested": deploy_auc - nested_auc,
            "accuracy_nested_rfecv": float(nested.get("accuracy", float("nan"))),
            "accuracy_deploy_schema": float(deploy.get("accuracy", float("nan"))),
            "nested_feature_count_median": nested_feature_count_median,
            "deploy_feature_count": deploy_feature_count,
            "loso_protocol_nested": PROTOCOL_NESTED_RFECV,
            "loso_protocol_deploy": PROTOCOL_DEPLOY_GLOBAL,
        })
    return sorted(rows, key=lambda r: r["loso_auc_nested_rfecv"], reverse=True)


def build_primary_endpoint_registry(
    config: dict,
    nested_results: dict[str, dict],
    deploy_results: dict[str, dict],
) -> dict[str, Any]:
    deploy_name = deploy_ensemble_model_name(config)
    best_nested = best_nested_model_name(nested_results) if nested_results else ""
    primary = resolve_primary_endpoint(config)

    if primary == ENDPOINT_ANOMALY_ENSEMBLE:
        guidance = (
            "Primary endpoint: trial-level LOSO OOF anomaly ensemble (Healthy-reference "
            "one-class screening). Supervised pathology-tier metrics below are secondary."
        )
    else:
        guidance = (
            "Report primary deployable ensemble AUC from loso_deploy_schema for API parity "
            "(RES-003). Cite best single-model loso_auc_nested_rfecv separately for "
            "unbiased screening benchmarks."
        )

    registry: dict[str, Any] = {
        "primary_endpoint": primary,
        "manuscript_guidance": guidance,
        "registered_endpoints": {},
    }

    if deploy_name in deploy_results:
        registry["registered_endpoints"][ENDPOINT_DEPLOY_ENSEMBLE] = {
            "model": deploy_name,
            "auc": float(deploy_results[deploy_name]["auc"]),
            "metric_source": "loso_deploy_schema",
            "feature_selection_protocol": PROTOCOL_DEPLOY_GLOBAL,
        }

    if best_nested and best_nested in nested_results:
        registry["registered_endpoints"][ENDPOINT_BEST_LOSO_NESTED] = {
            "model": best_nested,
            "auc": float(nested_results[best_nested]["auc"]),
            "metric_source": "loso_nested_rfecv",
            "feature_selection_protocol": PROTOCOL_NESTED_RFECV,
        }

    return registry


def write_deploy_loso_artifacts(
    metrics_dir: Path,
    gap_rows: list[dict[str, Any]],
    registry: dict[str, Any],
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if gap_rows:
        pd.DataFrame(gap_rows).to_csv(metrics_dir / "deploy_loso_gap.csv", index=False)
    if registry.get("primary_endpoint") != ENDPOINT_ANOMALY_ENSEMBLE:
        (metrics_dir / "primary_endpoint.json").write_text(
            json.dumps(registry, indent=2),
            encoding="utf-8",
        )


def write_anomaly_primary_artifacts(
    metrics_dir: Path,
    registry: dict[str, Any],
) -> None:
    """Write primary registry when anomaly ensemble is the manuscript endpoint."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "primary_endpoint.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def load_primary_endpoint_registry(metrics_dir: Path) -> dict[str, Any] | None:
    path = metrics_dir / "primary_endpoint.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_inference_checkpoint_name(config: dict, metrics_dir: Path) -> str:
    """Checkpoint used by predict stage and API (deploy ensemble by default)."""
    registry = load_primary_endpoint_registry(metrics_dir)
    if registry:
        deploy = registry.get("registered_endpoints", {}).get(ENDPOINT_DEPLOY_ENSEMBLE)
        if isinstance(deploy, dict) and deploy.get("model"):
            return str(deploy["model"])
    return deploy_ensemble_model_name(config)
