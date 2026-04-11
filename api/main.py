"""
FastAPI service for GaitGuard fall-risk prediction.

This API accepts uploaded IMU trial files, runs the same single-trial
preprocessing and feature extraction logic used by the training pipeline,
adapts the result into the patient-level feature schema expected by the saved
models, and returns a frontend-friendly prediction payload.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import secrets
import sys
import time
import warnings
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, cast

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.utils.validation import check_is_fitted

# Logging setup
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
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
from src.preprocessing.signal_processor import SignalProcessor  # type: ignore


def parse_cors_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [origin.strip() for origin in value.split(",") if origin.strip()]


cors_origins = parse_cors_origins(os.getenv("CORS_ORIGINS"))
allow_all_origins = cors_origins == ["*"]

if not cors_origins:
    cors_origins = ["http://localhost:8001", "http://127.0.0.1:8001"]
    allow_all_origins = False

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PIPELINE_ROOT / "configs" / "pipeline_config.yaml"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PIPELINE_ROOT / "results" / "checkpoints"))
ANOMALY_DIR = Path(os.getenv("ANOMALY_DIR", PIPELINE_ROOT / "results" / "anomaly_detection"))
PATIENT_FEATURES_PATH = Path(os.getenv("PATIENT_FEATURES_PATH", PIPELINE_ROOT / "data" / "features" / "patient_features.parquet"))
TRIAL_FEATURES_PATH = Path(os.getenv("TRIAL_FEATURES_PATH", PIPELINE_ROOT / "data" / "features" / "trial_features.parquet"))

REQUIRED_FILES = {
    "head_raw.csv": "head",
    "lower_back_raw.csv": "lower_back",
    "left_foot_raw.csv": "left_foot",
    "right_foot_raw.csv": "right_foot",
}
METADATA_FILE = "metadata.json"
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_TOTAL_SIZE_MB = float(os.getenv("MAX_TOTAL_SIZE_MB", "200"))

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
    with open(path, "rb") as fh:
        return pickle.load(fh)


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

    config = load_config()
    signal_processor = SignalProcessor(config)
    feature_extractor = FeatureExtractor(config)

    if not PATIENT_FEATURES_PATH.exists():
        raise RuntimeError(f"Missing feature schema: {PATIENT_FEATURES_PATH}")
    if not TRIAL_FEATURES_PATH.exists():
        raise RuntimeError(f"Missing trial feature schema: {TRIAL_FEATURES_PATH}")

    patient_df = pd.read_parquet(PATIENT_FEATURES_PATH)
    trial_df = pd.read_parquet(TRIAL_FEATURES_PATH)

    patient_meta_cols = ["participant_id", "cohort", "risk_label"]
    trial_meta_cols = [
        "trial_id", "participant_id", "cohort",
        "risk_label", "fall_probability", "laterality_biased",
    ]

    patient_feature_columns = (
        patient_df[[c for c in patient_df.columns if c not in patient_meta_cols]]
        .select_dtypes(include=np.number)
        .columns.tolist()
    )
    trial_feature_columns = (
        trial_df[[c for c in trial_df.columns if c not in trial_meta_cols]]
        .select_dtypes(include=np.number)
        .columns.tolist()
    )
    trial_feature_medians = cast(
        dict[str, float],
        trial_df[trial_feature_columns].median(numeric_only=True).fillna(0.0).to_dict(),
    )

    for model_name in ["ensemble", "lightgbm", "random_forest", "xgboost", "svm"]:
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if model_path.exists():
            models[model_name] = load_pickle(model_path)

    for model_name in ["isolation_forest", "lof", "one_class_svm"]:
        model_path = ANOMALY_DIR / f"{model_name}_model.pkl"
        scaler_path = ANOMALY_DIR / f"{model_name}_scaler.pkl"
        if model_path.exists() and scaler_path.exists():
            anomaly_models[model_name] = load_pickle(model_path)
            anomaly_scalers[model_name] = load_pickle(scaler_path)

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
    load_resources()
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(title="GaitGuard API", version="2.0.0", lifespan=lifespan)

if RATE_LIMITING_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_cohort_name(cohort: str | None) -> str:
    if not cohort:
        return "unknown"
    return str(cohort).strip().lower().replace(" ", "").replace("_", "")


def infer_sensor_name(filename: str) -> str:
    lowered = filename.lower()
    if "head" in lowered:
        return "head"
    if "back" in lowered:
        return "lower_back"
    if "left_foot" in lowered or "lfoot" in lowered:
        return "left_foot"
    if "right_foot" in lowered or "rfoot" in lowered:
        return "right_foot"
    raise HTTPException(status_code=400, detail=f"Unrecognized sensor file: {filename}")


def validate_csv_content(df: pd.DataFrame, sensor_name: str) -> None:
    if df.empty:
        raise HTTPException(status_code=400, detail=f"{sensor_name} CSV file is empty.")

    suspicious_patterns = ["<script", "javascript:", "onerror", "onload", "eval("]
    for col in df.columns:
        col_str = str(col).lower()
        for pattern in suspicious_patterns:
            if pattern in col_str:
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


def _safe_archive_read(archive: zipfile.ZipFile, safe_name: str, name_map: dict[str, str]) -> bytes:
    """Read from archive using only the validated safe name mapping."""
    original_name = name_map[safe_name]
    # Guard against path traversal: ensure the resolved path stays within archive
    resolved = Path(original_name).name.lower()
    if resolved != safe_name:
        raise HTTPException(status_code=400, detail="Invalid file path in ZIP archive.")
    return archive.read(original_name)


def parse_uploaded_files(files: list[UploadFile]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Too many files uploaded. Maximum 20 files allowed.")

    total_size = sum(f.size if f.size else 0 for f in files)
    if total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Total upload size exceeds maximum allowed size of {MAX_TOTAL_SIZE_MB}MB",
        )

    sensor_frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}

    # --- ZIP branch ---
    if len(files) == 1 and files[0].filename and files[0].filename.lower().endswith(".zip"):
        uploaded = files[0]
        if uploaded.size and uploaded.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB",
            )

        with zipfile.ZipFile(uploaded.file, "r") as archive:
            # Build a map of safe (basename only) → original archive name
            # Reject any entry whose basename differs from what we'd expect
            # to prevent path traversal attacks.
            name_map: dict[str, str] = {}
            for entry in archive.namelist():
                if entry.endswith("/"):
                    continue
                safe = Path(entry).name.lower()
                # Reject entries that try to escape via path components
                if Path(entry).name != Path(entry).parts[-1]:
                    continue
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
    uploads_by_name = {
        (upload.filename or "").lower(): upload
        for upload in files
        if upload.filename
    }
    required = [*REQUIRED_FILES.keys(), METADATA_FILE]
    missing = [n for n in required if n not in uploads_by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required files: {missing}")

    for upload in files:
        if not upload.filename:
            continue
        if upload.size and upload.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB",
            )
        lowered = upload.filename.lower()
        if lowered.endswith(".csv"):
            sensor_name = infer_sensor_name(lowered)
            df = pd.read_csv(upload.file)
            validate_csv_content(df, sensor_name)
            sensor_frames[sensor_name] = df
        elif lowered == METADATA_FILE:
            metadata = json.loads(upload.file.read().decode("utf-8"))

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
        clean = signal_processor._safe_filter(df.copy())
        clean = signal_processor._compute_resultant(clean)

        if sensor_name == "left_foot":
            clean = signal_processor._detect_gait_events(clean, "left")
        elif sensor_name == "right_foot":
            clean = signal_processor._detect_gait_events(clean, "right")
        elif sensor_name == "lower_back":
            clean = signal_processor._remove_gravity(clean)

        processed[sensor_name] = clean

    return processed


def extract_trial_features(processed: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> dict[str, Any]:
    if feature_extractor is None:
        raise RuntimeError("Feature extractor not initialized.")

    cohort = str(metadata.get("cohort", "unknown"))
    normalized_cohort = normalize_cohort_name(cohort)

    trial_features: dict[str, Any] = {
        "trial_id": metadata.get("trial_id", "uploaded_trial"),
        "participant_id": metadata.get("participant_id", "uploaded_participant"),
        "cohort": cohort,
        "fall_probability": float(COHORT_FALL_PROBABILITIES.get(normalized_cohort, 0.10)),
        "risk_label": 0,
        "laterality_biased": normalized_cohort in LATERALITY_BIASED_COHORTS,
    }

    lower_back = processed.get("lower_back")
    head = processed.get("head")
    left_foot = processed.get("left_foot")
    right_foot = processed.get("right_foot")

    if lower_back is not None:
        trial_features.update(feature_extractor._trunk_dynamics(lower_back))
        trial_features.update(feature_extractor._spectral_features(lower_back, prefix="lb"))

    if head is not None:
        trial_features.update(feature_extractor._trunk_dynamics(head, prefix="head"))

    if left_foot is not None and right_foot is not None:
        trial_features.update(feature_extractor._gait_cycle_features(left_foot, right_foot))
        trial_features.update(feature_extractor._asymmetry_features(left_foot, right_foot))

    return trial_features


def build_patient_feature_vector(trial_features: dict[str, Any]) -> pd.DataFrame:
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


def predict_risk(feature_vector: pd.DataFrame) -> tuple[dict[str, Any], str]:
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
    risk_probability = float(probabilities[1])
    risk_score = int(round(risk_probability * 100))

    if risk_score >= 70:
        risk_level = "high"
    elif risk_score >= 40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return (
        {
            "risk_score": risk_score,
            "risk_probability": risk_probability,
            "risk_level": risk_level,
            "confidence": float(max(probabilities)),
        },
        model_name,
    )


def predict_anomaly(trial_vector: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if not anomaly_models or not anomaly_scalers:
        return "Unknown", {"available": False}

    votes = 0
    details: dict[str, Any] = {"available": True, "methods": {}}

    for method_name, model in anomaly_models.items():
        scaler = anomaly_scalers[method_name]
        transformed = scaler.transform(trial_vector)
        label = int(model.predict(transformed)[0])
        score = float(-model.decision_function(transformed)[0])
        is_anomaly = label == -1
        votes += int(is_anomaly)
        details["methods"][method_name] = {
            "label": "anomaly" if is_anomaly else "normal",
            "score": score,
        }

    details["votes"] = votes
    return ("Detected" if votes >= 2 else "Normal"), details


def build_metadata_graph(prediction: dict[str, Any], anomaly_status: str, metadata: dict[str, Any]) -> dict[str, float]:
    confidence = float(prediction.get("confidence", 0.0)) * 100
    anomaly_score = 100 if anomaly_status == "Detected" else 20

    cohort = str(metadata.get("cohort", "unknown")).lower()
    cohort_map = {
        "healthy": 20,
        "post-surg": 60,
        "pd": 90,
        "cva": 85,
        "control": 20,
        "elderly": 70,
    }
    cohort_score = cohort_map.get(cohort, 50)
    trials_score = min(100, 1 * 20)  # currently always 1 trial

    return {
        "confidence": round(confidence, 2),
        "anomaly": anomaly_score,
        "cohort": cohort_score,
        "trials": trials_score,
    }


def build_response(
    metadata: dict[str, Any],
    trial_features: dict[str, Any],
    prediction: dict[str, Any],
    model_name: str,
    anomaly_status: str,
    anomaly_details: dict[str, Any],
) -> dict[str, Any]:
    graph_values = build_metadata_graph(prediction, anomaly_status, metadata)

    participant_id = metadata.get("participant_id") or metadata.get("patient_id") or "Uploaded Patient"
    trial_id = metadata.get("trial_id") or "Uploaded Trial"
    cohort = metadata.get("cohort", "Unknown")

    return {
        "success": True,
        "participant_id": participant_id,
        "trial_id": trial_id,
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "anomaly_status": anomaly_status,
        "graph_values": graph_values,
        "model_used": model_name,
        "metadata": {
            "patient_name": participant_id,
            "cohort": cohort,
            "confidence": f"{int(round(prediction['confidence'] * 100))}%",
            "anomaly": anomaly_status,
            "trials": "1",
        },
        "anomaly_details": anomaly_details,
        "request_id": secrets.token_hex(8),
        "generated_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "GaitGuard API is running",
        "models_loaded": sorted(models.keys()),
        "anomaly_models_loaded": sorted(anomaly_models.keys()),
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    try:
        return {
            "status": "healthy",
            "models_loaded": len(models),
            "anomaly_models_loaded": len(anomaly_models),
            "feature_count": len(patient_feature_columns),
            "api_version": "2.0.0",
        }
    except Exception:
        return {
            "status": "unhealthy",
            "models_loaded": 0,
            "anomaly_models_loaded": 0,
            "feature_count": 0,
            "api_version": "2.0.0",
        }


@app.post("/predict")
async def predict_fall_risk(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    # Apply rate limit when slowapi is available
    if RATE_LIMITING_AVAILABLE and limiter:
        await limiter._check_request_limit(request, predict_fall_risk, "10/minute")  # type: ignore[attr-defined]

    try:
        raw_sensor_frames, metadata = parse_uploaded_files(files)
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

        payload = build_response(
            metadata=metadata,
            trial_features=trial_features,
            prediction=prediction,
            model_name=model_name,
            anomaly_status=anomaly_status,
            anomaly_details=anomaly_details,
        )

        return JSONResponse(
            content=payload,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
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


# ---------------------------------------------------------------------------
# Entrypoint (local dev only — Render uses its own start command)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
