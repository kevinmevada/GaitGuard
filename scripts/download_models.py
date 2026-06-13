#!/usr/bin/env python3
"""
Download GaitGuard classification checkpoints and anomaly models from Hugging Face.

Set GAITGUARD_HF_REPO to a Hub model id (e.g. your-org/gaitguard-models) that
contains the file layout documented in docs/MODEL_CARD.md.

Train locally instead:
  cd fall_risk_pipeline && python main.py --stage train && python main.py --stage anomaly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
CHECKPOINT_DIR = PIPELINE_ROOT / "results" / "checkpoints"
ANOMALY_DIR = PIPELINE_ROOT / "results" / "anomaly_detection"

FEATURE_DIR = PIPELINE_ROOT / "data" / "features"
METRICS_DIR = PIPELINE_ROOT / "results" / "metrics"
HUB_REVISION_METADATA = CHECKPOINT_DIR / "hub_revision.json"

# Paths inside the Hub repo (must match docs/MODEL_CARD.md)
HUB_FILES: list[tuple[str, Path]] = [
    ("checkpoints/xgboost.pkl", CHECKPOINT_DIR / "xgboost.pkl"),
    ("checkpoints/lightgbm.pkl", CHECKPOINT_DIR / "lightgbm.pkl"),
    ("checkpoints/random_forest.pkl", CHECKPOINT_DIR / "random_forest.pkl"),
    ("checkpoints/svm.pkl", CHECKPOINT_DIR / "svm.pkl"),
    ("checkpoints/mlp.pkl", CHECKPOINT_DIR / "mlp.pkl"),
    ("checkpoints/ensemble.pkl", CHECKPOINT_DIR / "ensemble.pkl"),
    ("checkpoints/ensemble_stacking.pkl", CHECKPOINT_DIR / "ensemble_stacking.pkl"),
    ("checkpoints/checkpoint_manifest.json", CHECKPOINT_DIR / "checkpoint_manifest.json"),
    ("anomaly_detection/isolation_forest_model.pkl", ANOMALY_DIR / "isolation_forest_model.pkl"),
    ("anomaly_detection/isolation_forest_scaler.pkl", ANOMALY_DIR / "isolation_forest_scaler.pkl"),
    ("anomaly_detection/lof_model.pkl", ANOMALY_DIR / "lof_model.pkl"),
    ("anomaly_detection/lof_scaler.pkl", ANOMALY_DIR / "lof_scaler.pkl"),
    ("anomaly_detection/one_class_svm_model.pkl", ANOMALY_DIR / "one_class_svm_model.pkl"),
    ("anomaly_detection/one_class_svm_scaler.pkl", ANOMALY_DIR / "one_class_svm_scaler.pkl"),
    ("anomaly_detection/trial_feature_schema.json", ANOMALY_DIR / "trial_feature_schema.json"),
    ("anomaly_detection/checkpoint_manifest.json", ANOMALY_DIR / "checkpoint_manifest.json"),
    ("data/features/patient_features.parquet", FEATURE_DIR / "patient_features.parquet"),
    ("data/features/trial_features.parquet", FEATURE_DIR / "trial_features.parquet"),
    ("data/features/selected_features.json", FEATURE_DIR / "selected_features.json"),
    ("results/metrics/clinical_threshold.json", METRICS_DIR / "clinical_threshold.json"),
]

OPTIONAL_HUB_FILES = {
    "checkpoints/ensemble_stacking.pkl",
    "data/features/patient_features.parquet",
    "data/features/trial_features.parquet",
    "data/features/selected_features.json",
    "results/metrics/clinical_threshold.json",
}


def _verify_classification_manifest() -> None:
    """Verify downloaded checkpoints against checkpoint_manifest.json (DEP-006/SEC-009)."""
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.utils.checkpoint_io import assert_production_checkpoint_policy, load_checkpoint, manifest_path

    assert_production_checkpoint_policy()

    manifest = manifest_path(CHECKPOINT_DIR)
    if not manifest.is_file():
        print("WARN  no checkpoint manifest — skipping SHA-256 verification", file=sys.stderr)
        return

    for hub_path, local_path in HUB_FILES:
        if not hub_path.startswith("checkpoints/") or not local_path.suffix == ".pkl":
            continue
        if not local_path.is_file():
            continue
        try:
            load_checkpoint(local_path, manifest_dir=CHECKPOINT_DIR, require_manifest=True)
            print(f"VERIFY OK  {local_path.name}")
        except Exception as exc:
            print(f"VERIFY FAIL {local_path.name}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc


def _write_hub_revision_metadata(repo: str, revision: str) -> None:
    """Record pinned Hub revision for deploy audit (SEC-009)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": repo,
        "revision": revision,
        "pinned": revision.strip().lower() not in {"", "main", "master", "head"},
    }
    HUB_REVISION_METADATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GAITGUARD_HF_REPO", ""),
        help="Hugging Face model repo id (or set GAITGUARD_HF_REPO)",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("GAITGUARD_HF_REVISION", "main"),
        help="Git revision / tag on the Hub repo",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.utils.checkpoint_io import assert_production_hub_revision_policy

    try:
        assert_production_hub_revision_policy(args.revision)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.repo:
        print(
            "Error: Hugging Face repo not set.\n"
            "  export GAITGUARD_HF_REPO=your-org/gaitguard-models\n"
            "  python scripts/download_models.py --repo your-org/gaitguard-models\n\n"
            "Or train locally:\n"
            "  cd fall_risk_pipeline && python main.py --stage train\n"
            "  cd fall_risk_pipeline && python main.py --stage anomaly\n",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = 0, 0, 0
    for hub_path, local_path in HUB_FILES:
        optional = hub_path in OPTIONAL_HUB_FILES
        try:
            cached = hf_hub_download(
                repo_id=args.repo,
                filename=hub_path,
                revision=args.revision,
                repo_type="model",
            )
            local_path.write_bytes(Path(cached).read_bytes())
            print(f"OK  {hub_path} -> {local_path.relative_to(REPO_ROOT)}")
            ok += 1
        except Exception as exc:
            if optional:
                skipped += 1
                print(f"SKIP optional {hub_path} ({exc})")
                continue
            print(f"FAIL {hub_path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed.")
    if failed:
        return 1
    if ok == 0:
        print("No files downloaded — check repo id and MODEL_CARD.md layout.", file=sys.stderr)
        return 1
    _write_hub_revision_metadata(args.repo, args.revision)
    _verify_classification_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
