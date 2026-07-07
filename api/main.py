"""
FastAPI service for GaitGuard fall-risk prediction.

This API accepts uploaded IMU trial files, runs single-trial preprocessing and
feature extraction, then projects one trial into the patient-level feature schema
expected by saved models (degenerate aggregation: mean=trial value, std/range=0,
trend=NaN). Training used 260 participants with multi-trial aggregation (~1356
trials); inference is explicitly single-trial — see docs/inference_single_trial_limitation.md.
"""

from __future__ import annotations

import json
import logging
import asyncio
import os
import re
import secrets
import sys
import threading
import time
import warnings
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse


def _deployment_env() -> str:
    return os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()


def _is_production_deployment() -> bool:
    return _deployment_env() in ("production", "prod")


def _load_local_dotenv() -> None:
    """Load .env for local development only; production uses platform env (SEC-013)."""
    if _is_production_deployment():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; use system env vars


_load_local_dotenv()

import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.utils.validation import check_is_fitted
from starlette.middleware.base import BaseHTTPMiddleware

# Logging setup
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Training: 260 participants, 1356 trials, patient-level CV by participant_id.
# Inference: one uploaded trial → patient feature column names (not multi-trial stats).
INFERENCE_SCOPE = {
    "granularity": "single_trial",
    "model_training_granularity": "patient_level",
    "training_participants": 260,
    "training_trials": 1356,
    "patient_aggregation": "degenerate_single_trial",
}

INFERENCE_SCOPE_NOTE = (
    "Models were trained on patient-level rows (N≈260) with mean, std, range, and "
    "trend across multiple within-session trials (1356 trials total; CV grouped by "
    "participant_id). This request used one trial: values map to *_mean, *_std=0, "
    "*_range=0, *_trend=NaN. The score is trial-level screening, not a multi-trial "
    "patient summary."
)

CONFIDENCE_LIMITATION_NOTE = (
    "confidence is max(predicted class probability), not calibrated clinical "
    "certainty. Single-trial input omits cross-trial variability (std, range) and "
    "session trajectory (trend) present in training — interpret scores cautiously."
)

PAPER_INFERENCE_LIMITATION = (
    "Deployment inference accepted one uploaded trial per API request. To match "
    "the trained feature schema, trial values populated patient-level mean columns "
    "while standard deviation and range were set to zero and trend was undefined; "
    "scores therefore do not replicate full multi-trial patient aggregation used in "
    "training (260 participants, 1356 trials). Reported confidence reflects the "
    "model maximum class probability, not external clinical calibration."
)

DISPLAY_GAUGES_NOTE = (
    "Sidebar gauge bars (display_gauges) are UI-only. They are not SHAP values and "
    "must not be reported as clinical confidence in publications. anomaly_score is the "
    "primary continuous ensemble screening score (0–100); risk_score is a secondary "
    "supervised pathology-tier classifier output; trials is upload coverage vs mean "
    "training trials per participant, not a model score."
)

# Rate limiting / inference concurrency (SEC-008 / SEC-011)
_PREDICT_MAX_CONCURRENT_RAW = int(os.getenv("PREDICT_MAX_CONCURRENT", "2"))
PREDICT_MAX_CONCURRENT = min(4, max(2, _PREDICT_MAX_CONCURRENT_RAW))
PREDICT_REQUEST_TIMEOUT_SEC = max(10, int(os.getenv("PREDICT_REQUEST_TIMEOUT_SEC", "120")))
PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "10/minute")
_predict_semaphore = asyncio.Semaphore(PREDICT_MAX_CONCURRENT)


def _trust_proxy_headers() -> bool:
    """Trust X-Forwarded-For / X-Real-IP only behind a known reverse proxy."""
    explicit = os.getenv("TRUST_PROXY_HEADERS", "").lower()
    if explicit in ("1", "true", "yes"):
        return True
    if explicit in ("0", "false", "no"):
        return False
    return _is_production_deployment()


def _api_docs_enabled() -> bool:
    """
    SEC-010: disable Swagger/ReDoc/OpenAPI in production unless explicitly enabled.

    Set ENABLE_API_DOCS=true for short-lived debugging only.
    """
    explicit = os.getenv("ENABLE_API_DOCS", "").lower()
    if explicit in ("1", "true", "yes"):
        return True
    if explicit in ("0", "false", "no"):
        return False
    return not _is_production_deployment()


def _is_ui_response_path(path: str) -> bool:
    return path == "/" or path.startswith("/app") or path.startswith("/static")


def _security_headers_for_path(path: str) -> dict[str, str]:
    """SEC-010: baseline headers on every response; CSP tuned for bundled UI routes."""
    headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
    if _is_ui_response_path(path):
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
    else:
        headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if _is_production_deployment():
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for key, value in _security_headers_for_path(request.url.path).items():
            response.headers.setdefault(key, value)
        return response


# ---------------------------------------------------------------------------
# Lightweight, dependency-free request metrics (addresses "no API
# observability" — a bare log file was previously the only signal available
# for request volume, latency, or error rate). Intentionally NOT
# Prometheus-format (that would require the `prometheus_client` package,
# an extra dependency this API doesn't otherwise need) — a simple JSON
# counter exposed at GET /metrics, consistent with how /health already
# exposes operational state as JSON.
# ---------------------------------------------------------------------------
from collections import defaultdict as _defaultdict

_METRICS_LOCK = threading.Lock()
_METRICS_STATE: dict[str, dict[str, Any]] = _defaultdict(
    lambda: {"count": 0, "total_latency_s": 0.0, "error_count": 0, "status_counts": _defaultdict(int)}
)
_METRICS_STARTED_AT = time.time()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            key = f"{request.method} {request.url.path}"
            with _METRICS_LOCK:
                bucket = _METRICS_STATE[key]
                bucket["count"] += 1
                bucket["total_latency_s"] += elapsed
                bucket["status_counts"][str(status_code)] += 1
                if status_code >= 500:
                    bucket["error_count"] += 1
        return response


def _metrics_snapshot() -> dict[str, Any]:
    with _METRICS_LOCK:
        routes = {}
        for key, bucket in _METRICS_STATE.items():
            count = bucket["count"]
            routes[key] = {
                "request_count": count,
                "error_count": bucket["error_count"],
                "avg_latency_ms": round(1000 * bucket["total_latency_s"] / count, 2) if count else 0.0,
                "status_counts": dict(bucket["status_counts"]),
            }
    return {
        "uptime_seconds": round(time.time() - _METRICS_STARTED_AT, 1),
        "routes": routes,
    }


def rate_limit_client_key(request: Request) -> str:
    """Per-client slowapi key; uses leftmost X-Forwarded-For when proxy headers are trusted."""
    if _trust_proxy_headers():
        forwarded = (request.headers.get("X-Forwarded-For") or "").strip()
        if forwarded:
            client = forwarded.split(",")[0].strip()
            if client:
                return client
        real_ip = (request.headers.get("X-Real-IP") or "").strip()
        if real_ip:
            return real_ip

    client = getattr(request, "client", None)
    if client is not None and getattr(client, "host", None):
        return str(client.host)
    return "unknown"


# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=rate_limit_client_key)
    rate_limit_exception_handler = _rate_limit_exceeded_handler
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    limiter = None
    rate_limit_exception_handler = None
    RATE_LIMITING_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.features.feature_extractor import FeatureExtractor  # type: ignore
from src.models.anomaly_feature_schema import (  # type: ignore
    load_trial_feature_schema,
    validate_trial_columns_for_anomaly_scalers,
)
from src.evaluation.clinical_threshold import (  # type: ignore
    assign_risk_level,
    default_clinical_threshold,
    elevated_risk_probability,
    load_clinical_threshold_artifact,
)
from src.evaluation.research_disclaimers import (  # type: ignore
    RESEARCH_PROTOTYPE_DISCLAIMER,
    limitations_payload,
    screening_note,
)
from src.preprocessing.signal_processor import SignalProcessor  # type: ignore
from src.utils.checkpoint_io import CheckpointIntegrityError, assert_production_checkpoint_policy, load_checkpoint  # type: ignore
from src.evaluation.uncertainty import (  # type: ignore
    CalibrationArtifact,
    ConformalArtifact,
    apply_calibrator,
    conformal_prediction_set,
)


def parse_cors_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [origin.strip() for origin in value.split(",") if origin.strip()]


_DEFAULT_LOCAL_CORS_ORIGINS = (
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
)


def resolve_cors_origins(value: str | None) -> tuple[list[str], bool]:
    """Resolve allowed CORS origins and whether credentials may use a wildcard.

    Unset ``CORS_ORIGINS`` → localhost dev hosts only (denies arbitrary cross-origin sites).
    Explicit ``*`` is rejected and replaced with localhost defaults.
    """
    origins = parse_cors_origins(value)
    if origins == ["*"]:
        logger.warning(
            "CORS_ORIGINS='*' is not permitted; using localhost-only defaults. "
            "Set explicit frontend origin(s) in CORS_ORIGINS for production."
        )
        return list(_DEFAULT_LOCAL_CORS_ORIGINS), False
    if not origins:
        logger.warning(
            "CORS_ORIGINS is not set — all cross-origin browser requests WILL BE REJECTED "
            "from non-localhost frontends. Set CORS_ORIGINS=https://your-frontend-url "
            "before deploying."
        )
        return list(_DEFAULT_LOCAL_CORS_ORIGINS), False
    deploy_env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    if deploy_env not in ("production", "prod"):
        for local_origin in ("http://localhost:5500", "http://127.0.0.1:5500"):
            if local_origin not in origins:
                origins.append(local_origin)
    return origins, False


cors_origins, allow_all_origins = resolve_cors_origins(os.getenv("CORS_ORIGINS"))
if cors_origins == list(_DEFAULT_LOCAL_CORS_ORIGINS):
    logger.warning(
        "CORS_ORIGINS is not set — all cross-origin browser requests WILL BE REJECTED. "
        "Set CORS_ORIGINS=https://your-frontend-url before deploying."
    )

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PIPELINE_ROOT / "configs" / "pipeline_config.yaml"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PIPELINE_ROOT / "results" / "checkpoints"))
ANOMALY_DIR = Path(os.getenv("ANOMALY_DIR", PIPELINE_ROOT / "results" / "anomaly_detection"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", PIPELINE_ROOT / "results" / "metrics"))
ANOMALY_THRESHOLD_PATH = Path(
    os.getenv("ANOMALY_THRESHOLD_PATH", METRICS_DIR / "anomaly_threshold.json")
)
ANOMALY_CALIBRATION_PATH = Path(
    os.getenv("ANOMALY_CALIBRATION_PATH", ANOMALY_DIR / "deploy_calibration.json")
)
PATIENT_FEATURES_PATH = Path(os.getenv("PATIENT_FEATURES_PATH", PIPELINE_ROOT / "data" / "features" / "patient_features.parquet"))
TRIAL_FEATURES_PATH = Path(os.getenv("TRIAL_FEATURES_PATH", PIPELINE_ROOT / "data" / "features" / "trial_features.parquet"))
CLINICAL_THRESHOLD_PATH = Path(
    os.getenv(
        "CLINICAL_THRESHOLD_PATH",
        PIPELINE_ROOT / "results" / "metrics" / "clinical_threshold.json",
    )
)
CALIBRATION_ARTIFACT_PATH = Path(
    os.getenv("CALIBRATION_ARTIFACT_PATH", METRICS_DIR / "calibration_artifact.json")
)
CONFORMAL_ARTIFACT_PATH = Path(
    os.getenv("CONFORMAL_ARTIFACT_PATH", METRICS_DIR / "conformal_artifact.json")
)

REQUIRED_FILES = {
    "head_raw.csv": "head",
    "lower_back_raw.csv": "lower_back",
    "left_foot_raw.csv": "left_foot",
    "right_foot_raw.csv": "right_foot",
}
METADATA_FILE = "metadata.json"
ALLOWED_UPLOAD_NAMES = frozenset([*REQUIRED_FILES.keys(), METADATA_FILE])
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_TOTAL_SIZE_MB = float(os.getenv("MAX_TOTAL_SIZE_MB", "200"))
MAX_UNCOMPRESSED_ZIP_MB = float(
    os.getenv("MAX_UNCOMPRESSED_ZIP_MB", str(int(MAX_TOTAL_SIZE_MB)))
)
MAX_ZIP_COMPRESSION_RATIO = float(os.getenv("MAX_ZIP_COMPRESSION_RATIO", "100"))
API_KEY = os.getenv("GAITGUARD_API_KEY", "").strip()

COHORT_FALL_PROBABILITIES = {
    "healthy": 0.052,
    "hipoa": 0.285,
    "kneeoa": 0.241,
    "acl": 0.187,
    "pd": 0.673,
    "cva": 0.542,
    "cipn": 0.418,
    "ril": 0.389,
}
LATERALITY_BIASED_COHORTS = {"hipoa", "cva"}

# Global state populated during lifespan startup
config: dict[str, Any] = {}
signal_processor: SignalProcessor | None = None
feature_extractor: FeatureExtractor | None = None
models: dict[str, Any] = {}
patient_feature_columns: list[str] = []
trial_feature_columns: list[str] = []
anomaly_models: dict[str, Any] = {}
anomaly_scalers: dict[str, Any] = {}
trial_feature_medians: dict[str, float] = {}
expected_patient_feature_count: int | None = None
expected_trial_feature_count: int | None = None
anomaly_feature_schema: dict[str, Any] | None = None
patient_schema_mismatch: bool = False
trial_schema_mismatch: bool = False
anomaly_schema_message: str = ""
anomaly_threshold: dict[str, Any] = {}
anomaly_deploy_calibration: dict[str, Any] = {}
clinical_threshold: dict[str, Any] = {}
runtime_dependencies: dict[str, dict[str, Any]] = {}
calibration_artifact: CalibrationArtifact | None = None
conformal_artifact: ConformalArtifact | None = None

# Modules probed for /health; PyWavelets is mandatory at startup (fall_risk_pipeline/requirements.txt).
_DEPENDENCY_MODULES: dict[str, str] = {
    "PyWavelets": "pywt",
    "antropy": "antropy",
    "nolds": "nolds",
    "ahrs": "ahrs",
    "slowapi": "slowapi",
    "python-dotenv": "dotenv",
}
_STARTUP_REQUIRED_DEPENDENCIES = ("PyWavelets",)


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise RuntimeError("Pipeline config is invalid.")
    return loaded


def load_pickle(path: Path) -> Any:
    """Load a signed checkpoint; rejects files missing from the manifest."""
    try:
        return load_checkpoint(path, manifest_dir=path.parent, require_manifest=True)
    except CheckpointIntegrityError as exc:
        raise RuntimeError(
            f"Checkpoint integrity check failed for {path.name}: {exc}. "
            "Retrain models or refresh checkpoint_manifest.json from a trusted source."
        ) from exc


def _probe_runtime_dependencies() -> dict[str, dict[str, Any]]:
    """Import-check optional/required runtime libraries for health reporting."""
    status: dict[str, dict[str, Any]] = {}
    for label, module_name in _DEPENDENCY_MODULES.items():
        try:
            __import__(module_name)
            status[label] = {
                "available": True,
                "module": module_name,
                "required_at_startup": label in _STARTUP_REQUIRED_DEPENDENCIES,
            }
        except ImportError as exc:
            status[label] = {
                "available": False,
                "module": module_name,
                "required_at_startup": label in _STARTUP_REQUIRED_DEPENDENCIES,
                "error": str(exc),
            }
    return status


def _assert_startup_dependencies(dep_status: dict[str, dict[str, Any]]) -> None:
    """Fail fast before serving requests if mandatory libs are missing."""
    for label in _STARTUP_REQUIRED_DEPENDENCIES:
        entry = dep_status.get(label, {})
        if not entry.get("available"):
            if label == "PyWavelets":
                raise RuntimeError(
                    "PyWavelets required for wavelet feature computation. "
                    "Install fall_risk_pipeline/requirements.txt (PyWavelets>=1.4.0) and redeploy."
                )
            raise RuntimeError(
                f"{label} required. Install fall_risk_pipeline/requirements.txt and redeploy."
            )


def _verify_nonlinear_runtime_dependencies(
    patient_cols: list[str],
    trial_cols: list[str],
    dep_status: dict[str, dict[str, Any]] | None = None,
) -> None:
    """
    Fail fast if schema-required feature libs are missing.

    Without antropy/nolds, SampEn/DFA become NaN in feature extraction; without
    ahrs, Madgwick orientation columns are absent and orientation features
    silently drop out; without PyWavelets, wavelet features become NaN/zeroed.
    All can degrade API predictions.
    """
    all_cols = [str(c).lower() for c in (patient_cols + trial_cols)]
    needs_antropy = any("sampen" in c or "apen" in c for c in all_cols)
    needs_nolds = any("dfa" in c or "lyapunov" in c for c in all_cols)
    needs_ahrs = any(
        token in c
        for c in all_cols
        for token in ("tilt_", "pitch_", "roll_", "postural_sway")
    )
    needs_pywt = any("wavelet_" in c for c in all_cols)

    missing: list[str] = []
    dep_status = dep_status or runtime_dependencies

    def _require(label: str, module_name: str, needed: bool) -> None:
        if not needed:
            return
        if dep_status.get(label, {}).get("available"):
            return
        try:
            __import__(module_name)
        except ImportError:
            missing.append(label)

    _require("antropy", "antropy", needs_antropy)
    _require("nolds", "nolds", needs_nolds)
    _require("ahrs", "ahrs", needs_ahrs)
    _require("PyWavelets", "pywt", needs_pywt)

    if missing:
        raise RuntimeError(
            "Missing feature dependencies required by active schema: "
            f"{missing}. Install api requirements and redeploy."
        )


def is_model_fitted(model: Any) -> bool:
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def load_resources() -> None:
    global config, signal_processor, feature_extractor
    global models, patient_feature_columns, trial_feature_columns
    global anomaly_models, anomaly_scalers, trial_feature_medians
    global expected_patient_feature_count, expected_trial_feature_count
    global anomaly_feature_schema, patient_schema_mismatch, trial_schema_mismatch
    global anomaly_schema_message, clinical_threshold, runtime_dependencies
    global anomaly_threshold, anomaly_deploy_calibration
    global calibration_artifact, conformal_artifact

    config = load_config()
    deploy_env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    if deploy_env in ("production", "prod") and not API_KEY:
        raise RuntimeError(
            "GAITGUARD_API_KEY must be set when ENVIRONMENT=production (SEC-005)."
        )
    assert_production_checkpoint_policy()
    if deploy_env in ("production", "prod") and not RATE_LIMITING_AVAILABLE:
        raise RuntimeError(
            "slowapi must be installed when ENVIRONMENT=production (SEC-017)."
        )
    runtime_dependencies = _probe_runtime_dependencies()
    _assert_startup_dependencies(runtime_dependencies)
    loaded_threshold = load_clinical_threshold_artifact(CLINICAL_THRESHOLD_PATH)
    if loaded_threshold:
        clinical_threshold = loaded_threshold
        pc = loaded_threshold.get("primary_cutoff", {})
        logger.info(
            "Clinical cutoff loaded (Youden J): prob=%.3f sens=%.3f spec=%.3f [%s]",
            pc.get("probability", 0.5),
            pc.get("sensitivity", float("nan")),
            pc.get("specificity", float("nan")),
            pc.get("source", "unknown"),
        )
    else:
        clinical_threshold = default_clinical_threshold()
        logger.warning(
            "clinical_threshold.json not found at %s — using default 0.5; "
            "run `python main.py --stage evaluate` in fall_risk_pipeline.",
            CLINICAL_THRESHOLD_PATH,
        )

    if ANOMALY_THRESHOLD_PATH.is_file():
        with open(ANOMALY_THRESHOLD_PATH, encoding="utf-8") as fh:
            anomaly_threshold = json.load(fh)
        logger.info(
            "Anomaly screening cutoff loaded: prob=%.3f source=%s",
            anomaly_threshold.get("probability", 0.5),
            anomaly_threshold.get("source", "unknown"),
        )
    else:
        anomaly_threshold = {}

    calib_path = ANOMALY_CALIBRATION_PATH
    if not calib_path.is_file():
        calib_path = METRICS_DIR / "deploy_calibration.json"
    if calib_path.is_file():
        with open(calib_path, encoding="utf-8") as fh:
            anomaly_deploy_calibration = json.load(fh)
    else:
        anomaly_deploy_calibration = {}

    if CALIBRATION_ARTIFACT_PATH.is_file():
        try:
            calibration_artifact = CalibrationArtifact.from_json(CALIBRATION_ARTIFACT_PATH)
            logger.info("Post-hoc calibration artifact loaded from %s", CALIBRATION_ARTIFACT_PATH)
        except Exception:
            logger.exception(
                "Failed to load calibration artifact at %s — serving uncalibrated "
                "probabilities only.",
                CALIBRATION_ARTIFACT_PATH,
            )
            calibration_artifact = None
    else:
        calibration_artifact = None
        logger.info(
            "No calibration artifact found at %s — /predict will serve raw "
            "(uncalibrated) probabilities only. Run "
            "`python main.py --stage fit_uncertainty` in fall_risk_pipeline "
            "after `evaluate` to enable this.",
            CALIBRATION_ARTIFACT_PATH,
        )

    if CONFORMAL_ARTIFACT_PATH.is_file():
        try:
            conformal_artifact = ConformalArtifact.from_json(CONFORMAL_ARTIFACT_PATH)
            logger.info(
                "Split-conformal artifact loaded from %s (alpha=%.2f, q_hat=%.4f)",
                CONFORMAL_ARTIFACT_PATH,
                conformal_artifact.alpha,
                conformal_artifact.q_hat,
            )
        except Exception:
            logger.exception(
                "Failed to load conformal artifact at %s — /predict will omit "
                "prediction sets.",
                CONFORMAL_ARTIFACT_PATH,
            )
            conformal_artifact = None
    else:
        conformal_artifact = None
        logger.info(
            "No conformal artifact found at %s — /predict will omit distribution-free "
            "prediction sets. Run `python main.py --stage fit_uncertainty` in "
            "fall_risk_pipeline after `evaluate` to enable this.",
            CONFORMAL_ARTIFACT_PATH,
        )

    signal_processor = SignalProcessor(config)
    feature_extractor = FeatureExtractor(config)

    if not PATIENT_FEATURES_PATH.exists():
        raise RuntimeError(f"Missing feature schema: {PATIENT_FEATURES_PATH}")
    if not TRIAL_FEATURES_PATH.exists():
        raise RuntimeError(f"Missing trial feature schema: {TRIAL_FEATURES_PATH}")

    patient_df = pd.read_parquet(PATIENT_FEATURES_PATH)
    trial_df = pd.read_parquet(TRIAL_FEATURES_PATH)

    patient_meta_cols = [
        "participant_id", "cohort", "risk_label",
        "fall_probability", "laterality_biased", "n_trials",
    ]
    trial_meta_cols = [
        "trial_id", "participant_id", "cohort", "risk_label",
        "multiclass_label", "fall_probability", "laterality_biased",
    ]

    patient_feature_columns = (
        patient_df[[c for c in patient_df.columns if c not in patient_meta_cols]]
        .select_dtypes(include=np.number)
        .columns.tolist()
    )

    selected_path = PIPELINE_ROOT / "data" / "features" / "selected_features.json"
    if selected_path.exists():
        try:
            with open(selected_path, encoding="utf-8") as fh:
                selected_payload = json.load(fh)
            selected_names = selected_payload.get("features", [])
            if isinstance(selected_names, list) and selected_names:
                patient_feature_columns = [
                    c for c in patient_feature_columns if c in selected_names
                ]
                logger.info(
                    "Using %d selected patient-level features for inference",
                    len(patient_feature_columns),
                )
        except Exception as exc:
            logger.warning("Could not load selected_features.json: %s", exc)
    parquet_trial_columns = (
        trial_df[[c for c in trial_df.columns if c not in trial_meta_cols]]
        .select_dtypes(include=np.number)
        .columns.tolist()
    )
    trial_feature_columns = list(parquet_trial_columns)
    _verify_nonlinear_runtime_dependencies(
        patient_feature_columns,
        trial_feature_columns,
        dep_status=runtime_dependencies,
    )

    for model_name in ["ensemble", "lightgbm", "random_forest", "xgboost", "svm"]:
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if model_path.exists():
            models[model_name] = load_pickle(model_path)

    # Align runtime feature vectors to the fitted model input width.
    expected_patient_feature_count = None
    for preferred_name in ["ensemble", "lightgbm", "xgboost", "random_forest", "svm"]:
        model = models.get(preferred_name)
        if model is None:
            continue
        for attr_name in ("n_features_in_",):
            n_features = getattr(model, attr_name, None)
            if isinstance(n_features, int) and n_features > 0:
                expected_patient_feature_count = n_features
                break
        if expected_patient_feature_count is None and hasattr(model, "named_steps"):
            imputer = model.named_steps.get("imputer") if isinstance(model.named_steps, dict) else None
            n_features = getattr(imputer, "n_features_in_", None)
            if isinstance(n_features, int) and n_features > 0:
                expected_patient_feature_count = n_features
        if expected_patient_feature_count is not None:
            break

    for model_name in ["isolation_forest", "lof", "one_class_svm"]:
        model_path = ANOMALY_DIR / f"{model_name}_model.pkl"
        scaler_path = ANOMALY_DIR / f"{model_name}_scaler.pkl"
        if model_path.exists() and scaler_path.exists():
            anomaly_models[model_name] = load_pickle(model_path)
            anomaly_scalers[model_name] = load_pickle(scaler_path)

    anomaly_feature_schema = load_trial_feature_schema(ANOMALY_DIR)
    if anomaly_scalers:
        if anomaly_feature_schema is not None:
            trial_feature_columns = list(anomaly_feature_schema["feature_columns"])
        ok, anomaly_schema_message = validate_trial_columns_for_anomaly_scalers(
            trial_feature_columns,
            anomaly_scalers,
            anomaly_feature_schema,
        )
        trial_schema_mismatch = not ok
        expected_trial_feature_count = len(trial_feature_columns)
        missing_in_parquet = [
            c for c in trial_feature_columns if c not in parquet_trial_columns
        ]
        if missing_in_parquet:
            trial_schema_mismatch = True
            anomaly_schema_message = (
                f"{anomaly_schema_message}; parquet missing columns: "
                f"{missing_in_parquet[:5]}{'...' if len(missing_in_parquet) > 5 else ''}"
            )
    else:
        expected_trial_feature_count = len(trial_feature_columns)
        trial_schema_mismatch = False
        anomaly_schema_message = ""

    trial_feature_medians = cast(
        dict[str, float],
        trial_df.reindex(columns=trial_feature_columns)
        .median(numeric_only=True)
        .fillna(0.0)
        .to_dict(),
    )

    patient_schema_mismatch = (
        expected_patient_feature_count is not None
        and len(patient_feature_columns) != expected_patient_feature_count
    )

    if patient_schema_mismatch:
        logger.warning(
            "Patient feature schema mismatch: runtime=%d expected=%s",
            len(patient_feature_columns),
            expected_patient_feature_count,
        )
    if trial_schema_mismatch:
        logger.error(
            "Anomaly trial feature schema mismatch: %s (runtime n=%d, scaler n=%s)",
            anomaly_schema_message,
            len(trial_feature_columns),
            expected_trial_feature_count,
        )
    elif anomaly_scalers:
        logger.info(
            "Anomaly scalers aligned: n_features_in_=%s, columns=%d (Healthy-fit schema)",
            expected_trial_feature_count,
            len(trial_feature_columns),
        )

    if os.getenv("GAITGUARD_STRICT_SCHEMA", "").lower() in ("1", "true", "yes"):
        if patient_schema_mismatch:
            raise RuntimeError(
                f"Patient feature schema mismatch (expected {expected_patient_feature_count}, "
                f"got {len(patient_feature_columns)})."
            )
        if trial_schema_mismatch:
            raise RuntimeError(f"Anomaly trial feature schema mismatch: {anomaly_schema_message}")

    if not models:
        logger.warning(
            "No classification checkpoints in %s — API /predict will fail until you "
            "train (cd fall_risk_pipeline && python main.py --stage train) or download "
            "(GAITGUARD_HF_REPO=your-org/gaitguard-models python scripts/download_models.py). "
            "See docs/MODEL_CARD.md.",
            MODEL_DIR,
        )
    if not anomaly_models:
        logger.warning(
            "No anomaly models in %s — train with python main.py --stage anomaly or use "
            "scripts/download_models.py.",
            ANOMALY_DIR,
        )

    logger.info(
        "Resources loaded — models: %s | anomaly models: %s",
        sorted(models.keys()),
        sorted(anomaly_models.keys()),
    )


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if cors_origins == list(_DEFAULT_LOCAL_CORS_ORIGINS):
        logger.warning(
            "Startup: CORS limited to localhost — set CORS_ORIGINS for production frontends."
        )
    load_resources()
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_docs_enabled = _api_docs_enabled()
app = FastAPI(
    title="GaitGuard API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

if RATE_LIMITING_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)  # type: ignore[arg-type]

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_cohort_name(cohort: str | None) -> str:
    if not cohort:
        return "unknown"
    return str(cohort).strip().lower().replace(" ", "").replace("_", "")


def upload_basename(filename: str) -> str:
    """Lowercased basename only — rejects path components (SEC-007)."""
    if not filename or not str(filename).strip():
        raise HTTPException(status_code=400, detail="Upload filename is missing.")
    base = Path(filename).name.lower()
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid upload filename: {filename}")
    return base


def require_allowed_upload_name(filename: str) -> str:
    """Whitelist exact sensor/metadata basenames; reject substring guessing (SEC-007)."""
    base = upload_basename(filename)
    if base not in ALLOWED_UPLOAD_NAMES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_NAMES))
        raise HTTPException(
            status_code=400,
            detail=f"Unrecognized upload file '{filename}'. Allowed files: {allowed}",
        )
    return base


# IMU CSV column headers (validated before rename). API responses are JSON, not HTML.
_VALID_CSV_COLUMN_NAME = re.compile(r"^[A-Za-z0-9_.\- ]{1,64}$")


def validate_csv_content(df: pd.DataFrame, sensor_name: str) -> None:
    if df.empty:
        raise HTTPException(status_code=400, detail=f"{sensor_name} CSV file is empty.")

    for col in df.columns:
        col_str = str(col).strip()
        if not col_str or _VALID_CSV_COLUMN_NAME.fullmatch(col_str) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid column name detected in {sensor_name} file.",
            )

    if len(df.columns) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"{sensor_name} file has too many columns (possible attack).",
        )

    if len(df) > 100_000:
        raise HTTPException(
            status_code=400,
            detail=f"{sensor_name} file has too many rows (possible attack).",
        )


def _require_api_key(request: Request) -> None:
    """Optional shared-secret gate when GAITGUARD_API_KEY is set (SEC-001)."""
    if not API_KEY:
        return
    supplied = (request.headers.get("X-API-Key") or "").strip()
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip() if not supplied else supplied
    if not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _request_host(request: Request) -> str:
    host_header = (request.headers.get("host") or "").strip()
    if not host_header:
        return ""
    return host_header.split(":")[0].lower()


def _is_same_origin_ui_request(request: Request) -> bool:
    """
    True when the request appears to come from the bundled UI on this host (SEC-006).

    Browser UI must use POST /app/predict (same origin, no client-side secrets).
    Programmatic integrations use POST /predict with X-API-Key or an edge auth gateway.
    """
    host = _request_host(request)
    if not host:
        return False

    origin = (request.headers.get("origin") or "").strip()
    if origin:
        parsed = urlparse(origin)
        return bool(parsed.hostname and parsed.hostname.lower() == host)

    referer = (request.headers.get("referer") or "").strip()
    if not referer:
        return False
    parsed = urlparse(referer)
    if not parsed.hostname or parsed.hostname.lower() != host:
        return False
    path = parsed.path or ""
    return path == "/" or path.startswith("/app") or path.startswith("/static")


@asynccontextmanager
async def _predict_concurrency_slot():
    """
    Cap concurrent CPU-bound inference jobs (SEC-008 / SEC-011).

    Fast-fail with 503 when all slots are busy instead of unbounded ``to_thread`` queueing.
    """
    if _predict_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="Too many concurrent inference requests. Please retry shortly.",
        )
    await _predict_semaphore.acquire()
    try:
        yield
    finally:
        _predict_semaphore.release()


def _bounded_read(stream, max_bytes: int, *, label: str = "upload") -> bytes:
    """Stream-read with a hard byte cap (SEC-002: bypass of UploadFile.size)."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _is_safe_zip_member(name: str) -> bool:
    """Reject Zip Slip / absolute paths before reading archive entries (SEC-012)."""
    if not name or name.endswith("/") or "\0" in name:
        return False
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith(("/", "../")):
        return False
    return ".." not in path.parts


def _validate_zip_bomb_guard(archive: zipfile.ZipFile, compressed_size: int) -> None:
    """Reject archives with excessive uncompressed payload (SEC-003)."""
    total_uncompressed = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        total_uncompressed += int(info.file_size)
        limit = int(MAX_UNCOMPRESSED_ZIP_MB * 1024 * 1024)
        if total_uncompressed > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"ZIP uncompressed size exceeds {MAX_UNCOMPRESSED_ZIP_MB}MB "
                    "(possible zip bomb)"
                ),
            )
    if compressed_size > 0 and (total_uncompressed / compressed_size) > MAX_ZIP_COMPRESSION_RATIO:
        raise HTTPException(
            status_code=413,
            detail="ZIP compression ratio exceeds safe limit (possible zip bomb)",
        )


def _safe_archive_read(archive: zipfile.ZipFile, safe_name: str, name_map: dict[str, str]) -> bytes:
    """Read from archive using only the validated safe name mapping."""
    if safe_name not in name_map:
        raise HTTPException(status_code=400, detail=f"Missing ZIP member: {safe_name}")
    original_name = name_map[safe_name]
    if not _is_safe_zip_member(original_name):
        raise HTTPException(status_code=400, detail="Invalid file path in ZIP archive.")
    if Path(original_name).name.lower() != safe_name:
        raise HTTPException(status_code=400, detail="Invalid file path in ZIP archive.")
    return archive.read(original_name)


def parse_uploaded_files(files: list[UploadFile]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Too many files uploaded. Maximum 20 files allowed.")

    # Total size enforced via _bounded_read per file (SEC-004: do not trust UploadFile.size).

    sensor_frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}

    # --- ZIP branch ---
    if len(files) == 1 and files[0].filename and files[0].filename.lower().endswith(".zip"):
        uploaded = files[0]
        max_zip_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)
        zip_bytes = _bounded_read(uploaded.file, max_zip_bytes, label="ZIP archive")

        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as archive:
            if len(archive.infolist()) > 50:
                raise HTTPException(
                    status_code=413,
                    detail="ZIP archive has too many entries (possible zip bomb)",
                )
            _validate_zip_bomb_guard(archive, len(zip_bytes))
            # Build a map of safe (basename only) → original archive name
            # Reject any entry whose basename differs from what we'd expect
            # to prevent path traversal attacks.
            name_map: dict[str, str] = {}
            for entry in archive.namelist():
                if entry.endswith("/"):
                    continue
                if not _is_safe_zip_member(entry):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid file path in ZIP archive (path traversal rejected).",
                    )
                safe = Path(entry).name.lower()
                if safe not in ALLOWED_UPLOAD_NAMES:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"ZIP contains unrecognized file '{Path(entry).name}'. "
                            f"Allowed: {', '.join(sorted(ALLOWED_UPLOAD_NAMES))}"
                        ),
                    )
                if safe in name_map:
                    raise HTTPException(
                        status_code=400,
                        detail=f"ZIP contains duplicate file '{safe}'.",
                    )
                name_map[safe] = entry

            required = [*REQUIRED_FILES.keys(), METADATA_FILE]
            missing = [n for n in required if n not in name_map]
            if missing:
                raise HTTPException(status_code=400, detail=f"ZIP missing required files: {missing}")

            for file_name, sensor_name in REQUIRED_FILES.items():
                raw = _safe_archive_read(archive, file_name, name_map)
                df = pd.read_csv(BytesIO(raw))
                validate_csv_content(df, sensor_name)
                sensor_frames[sensor_name] = df

            raw_meta = _safe_archive_read(archive, METADATA_FILE, name_map)
            metadata = json.loads(raw_meta.decode("utf-8"))

        _validate_metadata(metadata)
        return sensor_frames, metadata

    # --- Individual files branch ---
    uploads_by_name: dict[str, UploadFile] = {}
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Upload filename is missing.")
        basename = require_allowed_upload_name(upload.filename)
        if basename in uploads_by_name:
            raise HTTPException(status_code=400, detail=f"Duplicate upload for {basename}")
        uploads_by_name[basename] = upload

    required = [*REQUIRED_FILES.keys(), METADATA_FILE]
    missing = [n for n in required if n not in uploads_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required files: {missing}")

    bytes_read = 0
    max_file_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)
    max_total_bytes = int(MAX_TOTAL_SIZE_MB * 1024 * 1024)

    for file_name, sensor_name in REQUIRED_FILES.items():
        upload = uploads_by_name[file_name]
        raw = _bounded_read(upload.file, max_file_bytes, label=f"File '{file_name}'")
        bytes_read += len(raw)
        if bytes_read > max_total_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload size exceeds maximum allowed size of {MAX_TOTAL_SIZE_MB}MB",
            )
        df = pd.read_csv(BytesIO(raw))
        validate_csv_content(df, sensor_name)
        sensor_frames[sensor_name] = df

    meta_upload = uploads_by_name[METADATA_FILE]
    raw_meta = _bounded_read(meta_upload.file, max_file_bytes, label=METADATA_FILE)
    bytes_read += len(raw_meta)
    if bytes_read > max_total_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Total upload size exceeds maximum allowed size of {MAX_TOTAL_SIZE_MB}MB",
        )
    metadata = json.loads(raw_meta.decode("utf-8"))

    _validate_metadata(metadata)
    return sensor_frames, metadata


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for field in ("participant_id", "trial_id"):
        if field not in metadata:
            raise HTTPException(status_code=400, detail=f"Missing required metadata field: {field}")


def normalize_sensor_dataframe(sensor_name: str, df: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    prefix_map = {
        "head": "he_",
        "lower_back": "lb_",
        "left_foot": "lf_",
        "right_foot": "rf_",
    }
    prefix = prefix_map[sensor_name]

    normalized = df.copy()
    normalized.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in normalized.columns
    ]

    rename_map: dict[str, str] = {}
    if "timestamp" in normalized.columns:
        rename_map["timestamp"] = "time"
    elif "time" not in normalized.columns:
        if "packetcounter" in normalized.columns:
            normalized["time"] = pd.to_numeric(normalized["packetcounter"], errors="coerce") / sampling_rate
        elif "packet_counter" in normalized.columns:
            normalized["time"] = pd.to_numeric(normalized["packet_counter"], errors="coerce") / sampling_rate
        else:
            normalized["time"] = np.arange(len(normalized), dtype=float) / sampling_rate

    candidate_map = {
        f"{prefix}acc_x": "acc_x", f"{prefix}acc_y": "acc_y", f"{prefix}acc_z": "acc_z",
        f"{prefix}gyr_x": "gyr_x", f"{prefix}gyr_y": "gyr_y", f"{prefix}gyr_z": "gyr_z",
        f"{prefix}mag_x": "mag_x", f"{prefix}mag_y": "mag_y", f"{prefix}mag_z": "mag_z",
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "acc_z",
        "gyr_x": "gyr_x", "gyr_y": "gyr_y", "gyr_z": "gyr_z",
        "mag_x": "mag_x", "mag_y": "mag_y", "mag_z": "mag_z",
    }
    for source, target in candidate_map.items():
        if source in normalized.columns:
            rename_map[source] = target

    normalized = normalized.rename(columns=rename_map)

    keep_cols = ["time", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z"]
    existing = [col for col in keep_cols if col in normalized.columns]
    if not {"acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"}.issubset(set(existing)):
        raise HTTPException(
            status_code=400,
            detail=f"{sensor_name} file is missing required accelerometer/gyroscope columns.",
        )

    normalized = normalized[existing].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.interpolate().bfill().ffill().dropna().reset_index(drop=True)

    if len(normalized) < 10:
        raise HTTPException(status_code=400, detail=f"{sensor_name} file is too short to analyze.")

    return normalized


def preprocess_sensor_data(sensor_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if signal_processor is None:
        raise RuntimeError("Signal processor not initialized.")

    processed: dict[str, pd.DataFrame] = {}
    for sensor_name, df in sensor_frames.items():
        processed[sensor_name] = signal_processor.process_sensor_dataframe(
            df.copy(), sensor_name
        )

    return processed


def extract_trial_features(processed: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> dict[str, Any]:
    if feature_extractor is None:
        raise RuntimeError("Feature extractor not initialized.")

    cohort = str(metadata.get("cohort", "unknown"))
    normalized_cohort = normalize_cohort_name(cohort)

    extraction_meta = {
        **metadata,
        "cohort": cohort,
        "fall_probability": float(COHORT_FALL_PROBABILITIES.get(normalized_cohort, 0.10)),
        "risk_label": 0,
        "laterality_biased": normalized_cohort in LATERALITY_BIASED_COHORTS,
    }
    return feature_extractor.extract_trial_features_from_processed(processed, extraction_meta)


def build_patient_feature_vector(trial_features: dict[str, Any]) -> pd.DataFrame:
    """
    Map one trial's features into the patient-level schema expected by checkpoints.

    Training aggregates multiple trials per participant (mean/std/range/trend).
    Here we use a degenerate mapping: mean=trial value, std=0, range=0, trend=NaN.
    """
    row: dict[str, Any] = {
        "participant_id": trial_features.get("participant_id", "uploaded_participant"),
        "cohort": trial_features.get("cohort", "unknown"),
        "risk_label": 0,
        "fall_probability": float(trial_features.get("fall_probability", 0.10)),
        "n_trials": 1.0,
        "laterality_biased": bool(trial_features.get("laterality_biased", False)),
    }

    excluded = {"trial_id", "participant_id", "cohort", "risk_label", "fall_probability", "laterality_biased"}
    for key, value in trial_features.items():
        if key in excluded:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            row[f"{key}_mean"] = float(value)
            row[f"{key}_std"] = 0.0
            row[f"{key}_range"] = 0.0
            row[f"{key}_trend"] = float("nan")

    feature_row = {col: row.get(col, np.nan) for col in patient_feature_columns}
    return pd.DataFrame([feature_row], columns=patient_feature_columns)


def build_trial_feature_vector(trial_features: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for col in trial_feature_columns:
        value = trial_features.get(col, trial_feature_medians.get(col, 0.0))
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            value = trial_feature_medians.get(col, 0.0)
        row[col] = float(value)
    return pd.DataFrame([row], columns=trial_feature_columns)


class UncertaintyFields(BaseModel):
    """Documents the optional additive fields _uncertainty_fields() may add
    to a /predict response. All Optional: every field is only present when
    the corresponding artifact (see fit_uncertainty pipeline stage) has been
    generated and loaded successfully."""

    calibrated_class_probabilities: list[float] | None = None
    calibrated_risk_probability: float | None = None
    conformal_prediction_set_class_indices: list[int] | None = None
    conformal_target_coverage: float | None = None
    conformal_note: str | None = None


def _uncertainty_fields(probabilities: np.ndarray) -> dict[str, Any]:
    """Best-effort additive fields from the optional calibration/conformal
    artifacts (see ``fit_uncertainty`` pipeline stage). Never raises — any
    shape mismatch or loading problem just omits these fields so the core
    prediction response is unaffected.
    """
    fields: dict[str, Any] = {}
    probs = np.asarray(probabilities, dtype=float).reshape(1, -1)

    if calibration_artifact is not None:
        try:
            if calibration_artifact.label_mode == "multiclass" and probs.shape[1] == calibration_artifact.n_classes:
                calibrated = apply_calibrator(calibration_artifact, probs)[0]
                fields["calibrated_class_probabilities"] = [float(p) for p in calibrated]
            elif calibration_artifact.label_mode == "binary" and probs.shape[1] >= 2:
                calibrated = apply_calibrator(calibration_artifact, probs[:, -1])[0]
                fields["calibrated_risk_probability"] = float(calibrated)
        except Exception:
            logger.exception("Calibration step failed for this request — omitting calibrated fields.")

    if conformal_artifact is not None:
        try:
            if probs.shape[1] > 1:
                sets = conformal_prediction_set(conformal_artifact, probs)
                fields["conformal_prediction_set_class_indices"] = sets[0]
                fields["conformal_target_coverage"] = 1.0 - conformal_artifact.alpha
                fields["conformal_note"] = (
                    "The listed class indices are guaranteed to contain the true class "
                    f"with approximately {100 * (1 - conformal_artifact.alpha):.0f}% probability "
                    "on data similar to the pipeline's LOSO evaluation set (split-conformal; "
                    "see docs/paper/methods.md). This is a coverage guarantee on the *set*, "
                    "not a statement about any single class's probability."
                )
        except Exception:
            logger.exception("Conformal step failed for this request — omitting prediction-set fields.")

    try:
        return UncertaintyFields(**fields).model_dump(exclude_none=True)
    except Exception:
        logger.exception(
            "Uncertainty fields failed their own schema validation — omitting "
            "all of them for this request rather than returning something malformed."
        )
        return {}


def predict_risk(feature_vector: pd.DataFrame) -> tuple[dict[str, Any], str]:
    if patient_schema_mismatch:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model artifact schema mismatch: retrain/export checkpoints with current "
                "feature pipeline before submission use."
            ),
        )

    if not models:
        raise HTTPException(status_code=500, detail="No trained prediction models are available.")

    model_name = ""
    model = None
    for candidate_name in ["ensemble", "lightgbm", "xgboost", "random_forest", "svm"]:
        candidate = models.get(candidate_name)
        if candidate is not None and is_model_fitted(candidate):
            model_name = candidate_name
            model = candidate
            break

    if model is None:
        raise HTTPException(status_code=500, detail="No fitted prediction models are available.")

    probabilities = model.predict_proba(feature_vector)[0]
    elevated_prob = float(elevated_risk_probability(probabilities.reshape(1, -1), config)[0])
    risk_score = int(round(elevated_prob * 100))

    primary = clinical_threshold.get("primary_cutoff", {})
    youden_prob = float(primary.get("probability", 0.5))
    risk_level = assign_risk_level(elevated_prob, youden_prob)

    note = screening_note(elevated_prob, youden_prob)

    payload = {
        "risk_score": risk_score,
        "risk_probability": elevated_prob,
        "risk_level": risk_level,
        "confidence": float(max(probabilities)),
        "clinical_cutoff_probability": youden_prob,
        "clinical_cutoff_risk_score": int(round(youden_prob * 100)),
        "above_clinical_cutoff": elevated_prob >= youden_prob,
        "screening_note": note,
        "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
    }
    payload.update(_uncertainty_fields(probabilities))

    return (payload, model_name)


def _normalise_deploy_anomaly_score(raw_score: float, method_name: str) -> float:
    """Map raw decision score to [0, 1] using deploy-time calibration ranges."""
    methods = anomaly_deploy_calibration.get("methods", {})
    calib = methods.get(method_name, {})
    lo = calib.get("min")
    hi = calib.get("max")
    if lo is None or hi is None or float(hi) - float(lo) < 1e-12:
        return 0.0
    normed = (float(raw_score) - float(lo)) / (float(hi) - float(lo))
    return float(max(0.0, min(1.0, normed)))


def predict_anomaly(trial_vector: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if trial_schema_mismatch:
        raise HTTPException(
            status_code=500,
            detail=(
                "Anomaly artifact schema mismatch: trial feature columns must match "
                "trial_feature_schema.json and each scaler's n_features_in_ (Healthy-fit). "
                f"{anomaly_schema_message} Retrain: python main.py --stage anomaly."
            ),
        )

    if not anomaly_models or not anomaly_scalers:
        return "Unknown", {"available": False}

    votes = 0
    details: dict[str, Any] = {"available": True, "methods": {}}
    norm_layers: list[float] = []

    for method_name, model in anomaly_models.items():
        scaler = anomaly_scalers[method_name]
        transformed = scaler.transform(trial_vector)
        label = int(model.predict(transformed)[0])
        score = float(-model.decision_function(transformed)[0])
        is_anomaly = label == -1
        votes += int(is_anomaly)
        norm_layers.append(_normalise_deploy_anomaly_score(score, method_name))
        details["methods"][method_name] = {
            "label": "anomaly" if is_anomaly else "normal",
            "score": score,
            "score_normalized": norm_layers[-1],
        }

    ensemble_norm = float(np.mean(norm_layers)) if norm_layers else 0.0
    anomaly_score = round(ensemble_norm * 100.0, 2)
    youden_prob = float(anomaly_threshold.get("probability", 0.5))
    vote_status = "Detected" if votes >= 2 else "Normal"
    youden_status = "Detected" if ensemble_norm >= youden_prob else "Normal"
    primary_status = youden_status if anomaly_threshold else vote_status

    details["votes"] = votes
    details["ensemble_score_normalized"] = ensemble_norm
    details["anomaly_score"] = anomaly_score
    details["youden_cutoff"] = youden_prob
    details["vote_status"] = vote_status
    details["screening_mode"] = "anomaly_ensemble"
    return primary_status, details


def build_display_gauges(
    prediction: dict[str, Any],
    anomaly_status: str,
    anomaly_details: dict[str, Any],
    *,
    n_trials_in_request: int = 1,
) -> dict[str, Any]:
    """UI sidebar bars — not SHAP; see DISPLAY_GAUGES_NOTE and returned schema."""
    confidence_pct = round(float(prediction.get("confidence", 0.0)) * 100, 2)
    risk_pct = float(prediction.get("risk_score", 0))

    anomaly_pct = float(anomaly_details.get("anomaly_score", 0.0))
    if anomaly_pct <= 0.0 and anomaly_details.get("available"):
        votes = int(anomaly_details.get("votes", 0))
        n_methods = len(anomaly_details.get("methods") or {})
        if n_methods > 0:
            anomaly_pct = round(100.0 * votes / n_methods, 2)
        elif anomaly_status == "Detected":
            anomaly_pct = 100.0
    elif anomaly_status == "Detected" and anomaly_pct <= 0.0:
        anomaly_pct = 100.0

    training_trials = float(INFERENCE_SCOPE["training_trials"])
    training_participants = float(INFERENCE_SCOPE["training_participants"])
    typical_trials_per_participant = training_trials / max(training_participants, 1.0)
    trial_coverage_pct = round(
        min(100.0, 100.0 * n_trials_in_request / max(typical_trials_per_participant, 1.0)),
        2,
    )

    values = {
        "anomaly_score": anomaly_pct,
        "anomaly": anomaly_pct,
        "risk_score": risk_pct,
        "confidence": confidence_pct,
        "trials": trial_coverage_pct,
    }

    return {
        "display_only": True,
        "not_shap": True,
        "description": DISPLAY_GAUGES_NOTE,
        "values": values,
        "sources": {
            "anomaly_score": "ensemble anomaly score × 100 (primary screening endpoint)",
            "anomaly": "alias of anomaly_score for legacy frontends",
            "risk_score": "secondary supervised pathology-tier classifier (0–100)",
            "confidence": "max predicted class probability × 100 (classifier)",
            "trials": (
                "n_trials_in_request vs mean training trials per participant "
                f"(~{typical_trials_per_participant:.1f}); coverage indicator only"
            ),
        },
    }


def build_response(
    metadata: dict[str, Any],
    trial_features: dict[str, Any],
    prediction: dict[str, Any],
    model_name: str,
    anomaly_status: str,
    anomaly_details: dict[str, Any],
) -> dict[str, Any]:
    n_trials = 1
    display_gauges = build_display_gauges(
        prediction,
        anomaly_status,
        anomaly_details,
        n_trials_in_request=n_trials,
    )
    gauge_values = display_gauges["values"]
    graph_values = {
        **gauge_values,
        "cohort": gauge_values.get("risk_score", 0),
    }

    participant_id = metadata.get("participant_id") or metadata.get("patient_id") or "Uploaded Patient"
    trial_id = metadata.get("trial_id") or "Uploaded Trial"
    cohort = metadata.get("cohort", "Unknown")
    anomaly_score = float(
        anomaly_details.get("anomaly_score", gauge_values.get("anomaly_score", 0))
    )

    return {
        "success": True,
        "participant_id": participant_id,
        "trial_id": trial_id,
        "anomaly_score": anomaly_score,
        "anomaly_status": anomaly_status,
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "single_trial_limitation": INFERENCE_SCOPE_NOTE,
        "confidence_limitation": CONFIDENCE_LIMITATION_NOTE,
        "display_gauges": display_gauges,
        "graph_values": graph_values,
        "model_used": model_name,
        "inference_scope": {
            **INFERENCE_SCOPE,
            "n_trials_in_request": n_trials,
            "note": INFERENCE_SCOPE_NOTE,
        },
        "clinical_threshold": {
            "primary_cutoff": clinical_threshold.get("primary_cutoff", {}),
            "clinical_screening_tools": clinical_threshold.get("clinical_screening_tools", {}),
            "borderline_moderate_band": clinical_threshold.get("borderline_moderate_band", {}),
            "anomaly_screening": anomaly_threshold,
            "this_request": {
                "elevated_probability": prediction.get("risk_probability"),
                "above_primary_cutoff": prediction.get("above_clinical_cutoff"),
                "anomaly_above_cutoff": anomaly_status == "Detected",
                "risk_level_rule": (
                    "high if elevated_probability ≥ Youden J cutoff; "
                    "moderate if in [0.5×Youden, Youden); low otherwise. "
                    "Not Morse/STRATIFY score equivalents."
                ),
            },
        },
        "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
        "screening_note": prediction.get("screening_note", ""),
        "limitations": {
            **limitations_payload(),
            "confidence": CONFIDENCE_LIMITATION_NOTE,
            "display_gauges": DISPLAY_GAUGES_NOTE,
            "clinical_threshold": (
                "Primary screening uses anomaly ensemble LOSO Youden cutoff "
                "(anomaly_threshold.json). Supervised risk level uses Youden J from "
                "pathology-tier LOSO (clinical_threshold.json)."
            ),
            "paper_methods": PAPER_INFERENCE_LIMITATION,
        },
        "metadata": {
            "patient_name": participant_id,
            "cohort": cohort,
            "confidence": f"{int(round(prediction['confidence'] * 100))}%",
            "confidence_meaning": "model max class probability (not clinical calibration)",
            "anomaly": anomaly_status,
            "anomaly_score": f"{int(round(anomaly_score))}%",
            "trials": str(n_trials),
            "inference_note": INFERENCE_SCOPE_NOTE,
        },
        "anomaly_details": anomaly_details,
        "request_id": secrets.token_hex(8),
        "generated_at": time.time(),
    }


def _run_prediction_pipeline(
    raw_sensor_frames: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """CPU-bound inference path (SEC-002) — run via asyncio.to_thread from /predict."""
    sampling_rate = float(metadata.get("sampling_rate", config["dataset"]["sampling_rate"]))

    normalized_frames = {
        sensor_name: normalize_sensor_dataframe(sensor_name, df, sampling_rate)
        for sensor_name, df in raw_sensor_frames.items()
    }
    processed_frames = preprocess_sensor_data(normalized_frames)
    trial_features = extract_trial_features(processed_frames, metadata)

    patient_vector = build_patient_feature_vector(trial_features)
    trial_vector = build_trial_feature_vector(trial_features)

    prediction, model_name = predict_risk(patient_vector)
    anomaly_status, anomaly_details = predict_anomaly(trial_vector)

    return build_response(
        metadata=metadata,
        trial_features=trial_features,
        prediction=prediction,
        model_name=model_name,
        anomaly_status=anomaly_status,
        anomaly_details=anomaly_details,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, Any]:
    if _is_production_deployment():
        return {
            "message": "GaitGuard API",
            "health": "/health",
            "ui_path": "/app",
        }
    ui = STATIC_DIR / "index.html"
    return {
        "message": "GaitGuard API is running",
        "models_loaded": sorted(models.keys()),
        "anomaly_models_loaded": sorted(anomaly_models.keys()),
        "ui_path": "/app" if ui.is_file() else None,
        "ui_predict_path": "/app/predict" if ui.is_file() else None,
        "programmatic_predict_path": "/predict",
        "static_mount": "/static" if STATIC_DIR.is_dir() else None,
        "docs_path": "/docs" if _docs_enabled else None,
    }


@app.get("/app")
async def serve_ui() -> FileResponse:
    """Bundled frontend served from api/static/."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=404,
            detail="UI not found — ensure api/static/index.html exists",
        )
    return FileResponse(index)


class HealthResponse(BaseModel):
    """Documented shape of GET /health.

    Every field is Optional because the endpoint has two code paths (normal
    and a fallback if the health check itself errors) that return different
    subsets of fields — see health_check() below. Making everything Optional
    here means this model documents the contract (and validates it) without
    risking a 500 on the exact request that's supposed to report "unhealthy".
    """

    status: str | None = None
    models_loaded: int | None = None
    anomaly_models_loaded: int | None = None
    patient_feature_count: int | None = None
    anomaly_trial_feature_count: int | None = None
    feature_count: int | None = None  # only present on the exception-fallback path
    expected_anomaly_n_features: int | None = None
    patient_schema_mismatch: bool | None = None
    trial_schema_mismatch: bool | None = None
    anomaly_schema_message: str | None = None
    dependencies: dict[str, Any] | None = None
    missing_required_dependencies: list[str] | None = None
    calibration_artifact_loaded: bool | None = None
    conformal_artifact_loaded: bool | None = None
    api_version: str | None = None


HealthResponse.model_rebuild()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    try:
        deps = runtime_dependencies or _probe_runtime_dependencies()
        missing_required = [
            name
            for name, info in deps.items()
            if info.get("required_at_startup") and not info.get("available")
        ]
        missing_inference = [
            name for name, info in deps.items() if not info.get("available")
        ]
        status = "healthy"
        if missing_required:
            status = "unhealthy"
        elif patient_schema_mismatch or trial_schema_mismatch or missing_inference:
            status = "degraded"
        return {
            "status": status,
            "models_loaded": len(models),
            "anomaly_models_loaded": len(anomaly_models),
            "patient_feature_count": len(patient_feature_columns),
            "anomaly_trial_feature_count": len(trial_feature_columns),
            "expected_anomaly_n_features": expected_trial_feature_count,
            "patient_schema_mismatch": patient_schema_mismatch,
            "trial_schema_mismatch": trial_schema_mismatch,
            "anomaly_schema_message": anomaly_schema_message or None,
            "dependencies": deps,
            "missing_required_dependencies": missing_required,
            "calibration_artifact_loaded": calibration_artifact is not None,
            "conformal_artifact_loaded": conformal_artifact is not None,
            "api_version": "2.0.0",
        }
    except Exception:
        return {
            "status": "unhealthy",
            "models_loaded": 0,
            "anomaly_models_loaded": 0,
            "feature_count": 0,
            "dependencies": runtime_dependencies or {},
            "api_version": "2.0.0",
        }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Basic request-volume/latency/error-rate counters, in memory since
    process start. Deliberately simple JSON (not Prometheus format, which
    would need an extra dependency this API doesn't otherwise require) —
    see the MetricsMiddleware definition above for what's tracked and why.
    Counters reset on process restart; for durable metrics, scrape this
    endpoint into an external time-series store rather than relying on it
    as the system of record.
    """
    return _metrics_snapshot()


@app.post("/app/predict")
async def predict_fall_risk_ui(
    request: Request, files: list[UploadFile] = File(...)
) -> JSONResponse:
    """
    Same-origin browser proxy for the bundled UI (SEC-006).

    Never embed GAITGUARD_API_KEY in static JS — this route validates Origin/Referer
    against the service Host. For cross-origin or programmatic access use POST /predict
    with X-API-Key, or place Cloudflare Access / Render auth in front of the service.
    """
    if not _is_same_origin_ui_request(request):
        raise HTTPException(
            status_code=403,
            detail=(
                "Bundled UI inference requires same-origin access via /app. "
                "Use POST /predict with X-API-Key for programmatic clients."
            ),
        )
    return await _execute_predict(files)


@app.post("/predict")
async def predict_fall_risk(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    _require_api_key(request)
    return await _execute_predict(files)


async def _execute_predict(files: list[UploadFile]) -> JSONResponse:
    try:
        raw_sensor_frames, metadata = await asyncio.to_thread(parse_uploaded_files, files)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload parsing failed")
        raise HTTPException(status_code=400, detail="Invalid upload.") from exc

    async with _predict_concurrency_slot():
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_run_prediction_pipeline, raw_sensor_frames, metadata),
                timeout=PREDICT_REQUEST_TIMEOUT_SEC,
            )

            return JSONResponse(
                content=payload,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate",
                    "Pragma": "no-cache",
                },
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Inference timed out after %ds (SEC-011)",
                PREDICT_REQUEST_TIMEOUT_SEC,
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Inference timed out after {PREDICT_REQUEST_TIMEOUT_SEC}s. "
                    "Try a smaller upload or retry later."
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unhandled error in /predict")
            if log_level == "DEBUG":
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(exc)}") from exc
            raise HTTPException(
                status_code=500,
                detail="Processing failed. Please verify uploaded files and try again.",
            ) from exc


if RATE_LIMITING_AVAILABLE and limiter:
    for route in app.routes:
        path = getattr(route, "path", None)
        if path in ("/predict", "/app/predict") and hasattr(route, "endpoint"):
            route.endpoint = limiter.limit(PREDICT_RATE_LIMIT)(route.endpoint)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Entrypoint (local dev only — Render uses its own start command)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
