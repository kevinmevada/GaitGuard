```md
# GaitGuard: Anomalous Gait Detection System

A machine learning system for detecting anomalous gait patterns from multi-sensor wearable IMU data.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Components](#components)
5. [Key Algorithms](#key-algorithms)
6. [Setup & Configuration](#setup--configuration)
7. [Running the System](#running-the-system)
8. [API Endpoints](#api-endpoints)
9. [Test Data](#test-data)
10. [File Structure](#file-structure)
11. [Technologies](#technologies)
12. [Usage Boundaries](#usage-boundaries)

---

## Project Overview

GaitGuard is a research-oriented anomalous gait detection system intended for experimentation, internal review, and reproducible ML workflows.

### Purpose

- Detect anomalous gait patterns relative to a training cohort of expected gaits
- Highlight gait irregularities and biomechanical pattern shifts
- Provide explainability via SHAP-style feature contribution analysis
- Support model development and anomaly benchmarking on open gait datasets
- Long-term aim: contribute to fall-prevention research (non-diagnostic)

### Key Statistics

- Status: research prototype with internal evaluation artifacts in this repository
- Dataset: 1,356 trials from 260 participants (8 open-data cohorts)
- Features: 41 temporal, spatial, spectral, and asymmetry indicators
- Models: 4-model ensemble (XGBoost, LightGBM, Random Forest, SVM)
- Anomaly Detection: 3 unsupervised methods (Isolation Forest, LOF, One-Class SVM)

### Open Data Cohorts

- Healthy (baseline gait patterns)
- Hip OA, Knee OA
- ACL injury, CIPN, RIL
- Parkinson's Disease, CVA/Stroke

Data Source: SpringerNature Figshare DOI: 10.6084/m9.figshare.28806086

---

## System Architecture

```text
GaitGuard System

Web Frontend (HTML5 + Three.js + Canvas)
- File upload (drag-drop, ZIP or individual files)
- Real-time signal visualization
- Interactive 3D human model with sensors
- Results dashboard (model score gauge, anomaly warnings)

(HTTP REST API)

FastAPI Backend (api/main.py)
- File validation and parsing
- Data normalization
- Preprocessing pipeline
- Feature extraction (41 features)
- Model ensemble inference
- Anomaly detection (3 methods)
- JSON response generation

Signal Processing | Feature Extraction | Models
- Filtering | Trunk dynamics | XGBoost
- Gravity removal | Gait metrics | LightGBM
- Gait detection | Spectral feats | RF
- Sensor fusion | Asymmetry | SVM
- | Variability | Ensemble

Model Score (0-100)
Model Confidence
Anomaly Status
SHAP Feature Contributions
```

---

## Data Flow

### Training Pipeline (Offline)

```text
Raw Dataset (Figshare)
[INGEST] Load CSVs, validate metadata
data/raw -> data/processed/signals/
[PREPROCESS] Filter, detect gait events, sensor fusion
data/processed/signals_clean/
[EDA] Exploratory analysis, visualize distributions
figures/eda/
[FEATURES] Extract 41 features per trial, aggregate to patient level
data/features/trial_features.parquet + patient_features.parquet
[TRAIN] Optuna hyperparameter tuning, train 4 models
results/checkpoints/*.pkl
[EVALUATE] Internal validation, SHAP analysis, and metrics reporting
results/metrics/, results/figures/
[ANOMALY] Unsupervised outlier detection (IF, LOF, One-Class SVM)
results/anomaly_detection/
[REPORT] Generate tables, figures, and summary artifacts
results/metrics/ieee_table.tex, pipeline_report.md
```

### Inference Pipeline (Online - API)

```text
User (Frontend)
[Upload IMU Files + Metadata]
(ZIP: 4 CSV + 1 JSON OR individual files)

API Validation
Check required files present
Validate CSV structure
Parse JSON metadata

[NORMALIZE] Convert column names, handle timestamps, interpolate
[PREPROCESS] Butterworth filter, gravity removal, gait event detection
[EXTRACT FEATURES] Compute 41 features from processed signals
[BUILD FEATURE VECTORS]
Patient-level: mean/std aggregation
Trial-level: median imputation for missing values

[MODEL INFERENCE]
Load ensemble (4 models from checkpoints/)
Soft voting probabilities
Return legacy fields: risk_score, risk_probability (model-derived scores)

[ANOMALY DETECTION]
Run 3 unsupervised methods (IF, LOF, One-Class SVM)
Majority voting: 2 votes or more -> Detected

[DERIVE DISPLAY FEATURES]
Extract 5 key biomechanics:
1. Stride Variability
2. Lateral Asymmetry
3. Cadence Instability
4. Head Acceleration RMS
5. Step Time Symmetry
```

---

## Components

### Pipeline (`fall_risk_pipeline/`)

Professional 10-stage ML pipeline:

| Stage | Module | Purpose |
|-------|--------|---------|
| Ingestion | `src/ingestion/data_loader.py` | Parse raw CSV files, validate metadata, create trial records |
| Preprocessing | `src/preprocessing/signal_processor.py` | Butterworth filtering, gravity removal, gait event detection, sensor fusion |
| EDA | Notebooks, visualizations | Exploratory analysis, signal distributions, outlier identification |
| Features | `src/features/feature_extractor.py` | Extract 41 temporal/spatial/spectral/asymmetry features |
| Training | `src/models/trainer.py` | Optuna hyperparameter tuning, train 4 models |
| Evaluation | `src/evaluation/evaluator.py`, `reporter.py` | Internal validation, SHAP analysis, metrics calculation |
| Prediction | `src/evaluation/predictions.py` | Generate predictions for new data |
| Anomaly | `src/models/anomaly_detector.py` | Unsupervised outlier detection (3 methods) |
| Report | `src/evaluation/reporter.py` | Generate tables and figures |

Config: `configs/pipeline_config.yaml`

### REST API (`api/main.py`)

FastAPI service for real-time inference:

Key functions:
- `parse_uploaded_files()` - Handle ZIP or individual file uploads
- `normalize_sensor_dataframe()` - Column mapping, timestamp handling, imputation
- `preprocess_sensor_data()` - Apply signal processing pipeline
- `extract_trial_features()` - Compute 41 features from processed signals
- `predict_risk()` - Ensemble inference with soft voting (legacy naming)
- `predict_anomaly()` - 3-method majority voting for outlier detection
- `derive_display_features()` - Extract 5 key biomechanics for UI
- `scale_display_shap()` - Z-score normalization for feature contributions

Endpoints:
- `POST /predict` - Upload trial data to get anomaly-focused model output
- `GET /` - API status
- `GET /health` - Health check
- `GET /docs` - Interactive Swagger UI

Runs at: `http://localhost:8001`

### Web Frontend (`Front_end/`)

Research dashboard with:
- `index.html` - Semantic HTML structure
- `main.js` - API communication, Three.js model, signal visualization
- `style.css` - UI design system

Features:
- Drag-drop file upload
- Real-time IMU signal visualization
- Interactive 3D human model with sensor placement
- Model performance table
- Pipeline stage animation
- Model score gauge (0-100 visual scale)
- Anomaly detection warnings
- Feature contribution cards
- Responsive design

---

## Key Algorithms

### Signal Processing

Butterworth Bandpass Filter
- Order: 4th
- Frequency range: 0.1 - 15 Hz
- Purpose: Remove noise, preserve gait-relevant frequencies

Gait Event Detection
- Method: Peak detection on vertical foot acceleration
- Purpose: Identify heel strikes, compute stride/step metrics

Gravity Removal
- Applied to: Lower back accelerometer
- Method: Remove DC (mean) component per trial
- Purpose: Isolate dynamic acceleration

Sensor Fusion (Madgwick Algorithm)
- Input: Accelerometer, gyroscope, magnetometer
- Output: Quaternion-based orientation
- Purpose: Fuse 9-axis IMU data into unified posture estimate

### Feature Engineering (41 Features)

1. Temporal/Gait Cycle
2. Spatial
3. Spectral
4. Trunk Dynamics
5. Asymmetry/Bilateral
6. Head/Postural

### Ensemble & Anomaly Methods

- Ensemble strategy: soft voting across XGBoost, LightGBM, Random Forest, SVM
- Unsupervised anomaly methods: Isolation Forest, LOF, One-Class SVM
- Voting: majority (2/3 methods flag anomaly) -> `Detected`

### Explainability (SHAP-style)

- Global: average feature contribution ranking
- Local: per-trial feature contributions
- Display normalization via z-score mapping to 0-100 contribution scale

---

## Setup & Configuration

### Environment Setup

```bash
cd c:\Users\mevad\Desktop\GI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r fall_risk_pipeline/requirements.txt
pip install -r api/requirements.txt
```

### Configuration File

Primary config: `fall_risk_pipeline/configs/pipeline_config.yaml`

### Dataset Organization

```text
fall_risk_pipeline/data/
  raw/
  processed/
  features/
```

Download Dataset:
1. Visit https://springernature.figshare.com/articles/dataset/.../28806086
2. Download ZIP
3. Extract into `fall_risk_pipeline/data/raw/`

---

## Running the System

### Option 1: Run Training Pipeline

```bash
cd fall_risk_pipeline
python main.py --config configs/pipeline_config.yaml
```

### Option 2: Run API Server

```bash
cd api
python start_api.py
```

Or:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Option 3: Open Frontend

```bash
start http://localhost:8001
```

---

## API Endpoints

### `POST /predict`

Upload trial data and receive anomaly-focused model output.

**Note:** Response fields currently include legacy names (`risk_score`, `risk_probability`) for compatibility.

### `GET /`

API status.

### `GET /health`

Health check.

### `GET /docs`

Interactive Swagger docs.

---

## Test Data

Location: `test/P1-P6/`

Each test dataset contains:
- 4 sensor CSV files (`head_raw.csv`, `lower_back_raw.csv`, `left_foot_raw.csv`, `right_foot_raw.csv`)
- 1 `metadata.json` file

Use with:

```bash
cd api
python start_api.py
```

---

## File Structure

Key directories:

- `api/` - FastAPI inference service
- `fall_risk_pipeline/` - training/evaluation pipeline
- `Front_end/` - frontend dashboard
- `test/` - sample upload bundles
- `examples/` - example sensor files

---

## Technologies

### Backend

- FastAPI, Uvicorn
- scikit-learn, XGBoost, LightGBM
- Optuna
- Pandas, NumPy, SciPy
- SHAP

### Frontend

- HTML5, CSS3
- Three.js
- Canvas API
- Fetch API

---

## Usage Boundaries

- GaitGuard is for research and informational purposes only.
- Outputs are not medical advice, diagnosis, or treatment guidance.
- The system does not establish clinical validity or provide clinical decision support.
- The long-term objective is to contribute evidence and tools for fall-prevention research.

---

## License

This project is intended for research and educational purposes.

---

## Citation

If you use this system in your research, cite the original dataset:

SpringerNature Figshare DOI: 10.6084/m9.figshare.28806086
```
