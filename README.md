# GaitGuard: Clinical Fall Risk Prediction System

A machine learning system for exploring fall-risk prediction from multi-sensor wearable IMU data.

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

---

## Project Overview

GaitGuard is a research-oriented fall risk prediction system intended for experimentation, review, and further validation.

### Purpose

- Predict fall probability (0-100 score) for individual patients
- Identify gait abnormalities and biomechanical risk factors
- Support clinical decision-making for fall prevention
- Provide explainability via SHAP feature importance analysis
- Detect anomalous gait patterns using unsupervised learning

### Key Statistics

- Status: research prototype with internal evaluation artifacts in this repository
- Dataset: 1,356 trials from 260 participants (8 clinical cohorts)
- Features: 41 temporal, spatial, spectral, and asymmetry indicators
- Models: 4-model ensemble (XGBoost, LightGBM, Random Forest, SVM)
- Anomaly Detection: 3 unsupervised methods (Isolation Forest, LOF, One-Class SVM)

### Clinical Cohorts

- Healthy (5.2% fall probability)
- Hip OA (28.5%), Knee OA (24.1%)
- ACL injury (18.7%), CIPN (41.8%), RIL (38.9%)
- Parkinson's Disease (67.3%), CVA/Stroke (54.2%)

Data Source: SpringerNature Figshare DOI: 10.6084/m9.figshare.28806086

---

## System Architecture

```
GaitGuard System

Web Frontend (HTML5 + Three.js + Canvas)
- File upload (drag-drop, ZIP or individual files)
- Real-time signal visualization
- Interactive 3D human model with sensors
- Results dashboard (risk gauge, anomaly warnings)

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

Risk Score (0-100)
Risk Level (High/Mod/Low)
Confidence Score
Anomaly Status
SHAP Feature Importance
```

---

## Data Flow

### Training Pipeline (Offline)

```
Raw Dataset (Figshare)
[INGEST] Load CSVs, validate metadata
data/raw to data/processed/signals/
[PREPROCESS] Filter, detect gait events, sensor fusion
data/processed/signals_clean/
[EDA] Exploratory data analysis, visualize distributions
figures/eda/
[FEATURES] Extract 41 features per trial, aggregate to patient level
data/features/trial_features.parquet + patient_features.parquet
[TRAIN] Optuna hyperparameter tuning, train 4 models
results/checkpoints/*.pkl
[EVALUATE] Internal validation, SHAP analysis, and metrics reporting
results/metrics/, results/figures/
[ANOMALY] Unsupervised outlier detection (IF, LOF, One-Class SVM)
results/anomaly_detection/
[REPORT] Generate IEEE-ready tables, figures, results summary
results/metrics/ieee_table.tex, pipeline_report.md
```

### Inference Pipeline (Online - API)

```
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

[PREDICT RISK]
Load ensemble (4 models from checkpoints/)
Soft voting with calibrated probabilities
Return risk_score (0-100), risk_probability (0-1)

[ANOMALY DETECTION]
Run 3 unsupervised methods (IF, LOF, One-Class SVM)
Majority voting: 2 votes or more to Detected

[DERIVE DISPLAY FEATURES]
Extract 5 key biomechanics:
1. Stride Variability
2. Lateral Asymmetry
3. Cadence Instability
4. Head Acceleration RMS
5. Step Time Symmetry

[SHAP NORMALIZATION]
Z-score each feature against population
Map 2 sigma to 0-100% contribution
Invert symmetry (higher symmetry = lower risk)

[RESPONSE JSON]
{
  "success": true,
  "risk_score": 65,
  "risk_level": "moderate",
  "risk_probability": 0.65,
  "confidence": 0.89,
  "anomaly_status": "Normal",
  "features": {...},
  "graph_values": {...},
  "shap_values": [...],
  "model_used": "ensemble"
}

Frontend (Display Results)
Risk gauge (visual 0-100 scale)
Anomaly warnings
Feature cards
Model confidence
Cohort comparison
```

---

## Components

### Fall Risk Pipeline (fall_risk_pipeline/)

Professional 10-stage ML pipeline:

| Stage | Module | Purpose |
|-------|--------|---------|
| Ingestion | src/ingestion/data_loader.py | Parse raw CSV files, validate metadata, create trial records |
| Preprocessing | src/preprocessing/signal_processor.py | Butterworth filtering, gravity removal, gait event detection, sensor fusion |
| EDA | Notebooks, visualizations | Exploratory analysis, signal distributions, outlier identification |
| Features | src/features/feature_extractor.py | Extract 41 temporal/spatial/spectral/asymmetry features |
| Training | src/models/trainer.py | Optuna Bayesian hyperparameter tuning, train 4 models |
| Evaluation | src/evaluation/evaluator.py, reporter.py | Internal validation, SHAP analysis, metrics calculation |
| Prediction | src/evaluation/predictions.py | Generate predictions for new data |
| Anomaly | src/models/anomaly_detector.py | Unsupervised outlier detection (3 methods) |
| Report | src/evaluation/reporter.py | Generate IEEE tables, publication-ready figures |

Config: configs/pipeline_config.yaml - sampling rate, filter specs, model hyperparams, feature list

### REST API (api/main.py)

FastAPI service for real-time inference:

Key Functions:
- parse_uploaded_files() - Handle ZIP or individual file uploads
- normalize_sensor_dataframe() - Column mapping, timestamp handling, data imputation
- preprocess_sensor_data() - Apply signal processing pipeline
- extract_trial_features() - Compute 41 features from processed signals
- predict_risk() - Ensemble inference with soft voting
- predict_anomaly() - 3-method majority voting for outlier detection
- derive_display_features() - Extract 5 key biomechanics for UI
- scale_display_shap() - Z-score normalization for SHAP-style contributions

Endpoints:
POST /predict - Upload trial data to get risk prediction
GET / - API status
GET /health - Health check
GET /docs - Interactive Swagger UI

Runs at: http://localhost:8001

### Web Frontend (Front_end/)

Clinical dashboard with:
- index.html - Semantic HTML structure
- main.js - API communication, Three.js 3D model, signal visualization
- style.css - Apple Design System aesthetics (dark theme, glassmorphism)

Features:
- Drag-drop file upload
- Real-time IMU signal visualization (Canvas API)
- Interactive 3D human model with sensor placement
- Model performance table (all 4 models + ensemble)
- Pipeline stage animation
- Risk gauge (0-100 visual scale, color-coded)
- Anomaly detection warnings
- Feature importance cards
- Responsive design (mobile-friendly)

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

1. Temporal/Gait Cycle (8 features)
- Stride time (mean, std, CV)
- Cadence, stride length
- Stance/swing ratio, double support ratio

2. Spatial (5 features)
- Step length (mean, std, asymmetry)
- Gait speed, step width

3. Spectral (6 features)
- Dominant frequency (Welch periodogram)
- Spectral entropy
- Band power (0-1 Hz, 1-3 Hz)
- Spectral centroid

4. Trunk Dynamics (12 features)
- RMS acceleration (AP, ML, vertical, resultant)
- Jerk (mean, max)
- Lyapunov exponent (gait chaos)
- Approximate entropy

5. Asymmetry/Bilateral (10 features)
- Stride time asymmetry (L/R)
- Swing time asymmetry
- Step length asymmetry
- RMS asymmetry across sensors
- Lateral variability

6. Head/Postural (5 features)
- Head acceleration RMS
- Head tilt angle
- Postural stability index

### Model Ensemble

Individual Classifiers (Bayesian hyperparameter tuning via Optuna):
- XGBoost: tree-based baseline for internal comparison
- LightGBM: gradient boosting baseline for internal comparison
- Random Forest: bagged tree baseline for internal comparison
- SVM: nonlinear kernel baseline for internal comparison

Ensemble Strategy: Soft voting with calibrated probabilities
- Predicted Probabilities = (p_xgb + p_lgb + p_rf + p_svm) / 4
- Historical internal benchmark: retained for reference only until validation is rerun

### Anomaly Detection

3 Independent Methods (unsupervised outlier scoring):

1. Isolation Forest
   - Anomaly score: Depth in isolation tree
   - Threshold: Default (anomaly_score > 0.5)

2. Local Outlier Factor (LOF)
   - Anomaly score: Ratio of local density to neighbors
   - Threshold: LOF > 1.5 (compared to neighbors)

3. One-Class SVM
   - Anomaly score: Distance to hyperplane
   - Threshold: Decision function < 0

Voting: majority (2/3 methods flag anomaly) to Detected

### Explainability (SHAP)

SHAP TreeExplainer for model interpretability:
- Global: Average |SHAP value| across all samples to feature importance ranking
- Local: Per-patient feature contributions to that patient's prediction
- Visualization: Waterfall plots, force plots, dependence plots

Display Features Normalization:
```
z-score = (feature_value - population_mean) / population_std
mapping:  z = -2 to 0%   (2 sigma below mean)
          z =  0 to 50%  (at population mean)
          z = +2 to 100% (2 sigma above mean)

For Step Time Symmetry: inverted ratio (1 - ratio)
Rationale: Higher symmetry = lower fall risk contribution
```

---

## Setup & Configuration

### Environment Setup

```bash
Navigate to workspace
cd c:\Users\mevad\Desktop\GI

Create virtual environment (if not exists)
python -m venv venv

Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

Install dependencies
pip install -r fall_risk_pipeline/requirements.txt
pip install -r api/requirements.txt
```

### Configuration File

fall_risk_pipeline/configs/pipeline_config.yaml:

```yaml
dataset:
  sampling_rate: 100
  sensors:
    - head
    - lower_back
    - left_foot
    - right_foot
  sensor_columns:
    - time
    - acc_x
    - acc_y
    - acc_z
    - gyr_x
    - gyr_y
    - gyr_z
    - mag_x
    - mag_y
    - mag_z
  min_trial_length: 10  # seconds

preprocessing:
  lowpass_cutoff: 15    # Hz
  highpass_cutoff: 0.1  # Hz
  butter_order: 4
  madgwick_beta: 0.1

models:
  enabled:
    - xgboost
    - lightgbm
    - random_forest
    - svm
  training:
    cv_folds: 5
    optuna_trials: 50
    timeout: 1200  # seconds
  ensemble:
    method: soft_voting
    top_models: 3

anomaly_detection:
  methods:
    - isolation_forest
    - lof
    - one_class_svm
  voting_threshold: 2  # majority

output:
  shap_samples: 200
  figure_dpi: 300
  figure_format: pdf
```

### Dataset Organization

```
fall_risk_pipeline/data/
raw/
  healthy/
  neuro/
  ortho/
  [download dataset here]
processed/
  signals/
  signals_clean/
  trial_metadata.csv
features/
  trial_features.parquet
  patient_features.parquet
```

Download Dataset:
1. Visit: https://springernature.figshare.com/articles/dataset/.../28806086
2. Download ZIP
3. Extract to fall_risk_pipeline/data/raw/

### Model Checkpoints

Trained models are loaded from: fall_risk_pipeline/results/checkpoints/

```
results/checkpoints/
ensemble.pkl           # Soft voting ensemble
xgboost.pkl           # XGBoost classifier
lightgbm.pkl          # LightGBM classifier
random_forest.pkl     # Random Forest classifier
svm.pkl               # SVM classifier

results/anomaly_detection/
isolation_forest_model.pkl
isolation_forest_scaler.pkl
lof_model.pkl
lof_scaler.pkl
one_class_svm_model.pkl
one_class_svm_scaler.pkl
```

---

## Running the System

### Option 1: Run Training Pipeline (Offline)

```bash
cd fall_risk_pipeline

Run all stages
python main.py --config configs/pipeline_config.yaml

Or run specific stages
python main.py --stage ingest
python main.py --stage preprocess
python main.py --stage eda
python main.py --stage features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage anomaly
python main.py --stage report
```

Output: Trained models in results/checkpoints/, metrics in results/metrics/

### Option 2: Run API Server (Real-time Inference)

```bash
Method 1: Using provided startup script
cd api
python start_api.py

Method 2: Direct run
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Method 3: Production deployment
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

API available at: http://localhost:8001
Swagger Docs: http://localhost:8001/docs
ReDoc: http://localhost:8001/redoc

### Option 3: Open Frontend Dashboard

```bash
Open the web UI (assuming local server)
start http://localhost:8001
OR
Open to local file (limited functionality)
start Front_end/index.html
```

---

## API Endpoints

### POST /predict

Upload trial data and get risk prediction

Request:
```
POST /predict
Content-Type: multipart/form-data

files:
  - head_raw.csv
  - lower_back_raw.csv
  - left_foot_raw.csv
  - right_foot_raw.csv
  - metadata.json
```

Metadata JSON:
```json
{
  "participant_id": "P001",
  "trial_id": "trial_001",
  "cohort": "healthy",
  "sampling_rate": 100,
  "age": 65,
  "gender": "M"
}
```

Response (200 OK):
```json
{
  "success": true,
  "participant_id": "P001",
  "trial_id": "trial_001",
  "risk_score": 35,
  "risk_level": "low",
  "risk_probability": 0.35,
  "confidence": 0.92,
  "anomaly_status": "Normal",
  "features": {
    "stride_variability": 0.08,
    "lateral_asymmetry": 0.12,
    "cadence_instability": 0.05,
    "head_accel_rms": 2.3,
    "step_time_symmetry": 0.95
  },
  "graph_values": {
    "stride_variability": 42,
    "lateral_asymmetry": 38,
    "cadence_instability": 35,
    "head_accel_rms": 45,
    "step_time_symmetry": 88
  },
  "shap_values": [42, 38, 35, 45, 88],
  "model_used": "ensemble",
  "metadata": {
    "patient_name": "P001",
    "trial_id": "trial_001",
    "cohort": "healthy",
    "confidence": "92%",
    "anomaly": "Normal"
  },
  "anomaly_details": {
    "available": true,
    "methods": {
      "isolation_forest": {"label": "normal", "score": 0.2},
      "lof": {"label": "normal", "score": 0.3},
      "one_class_svm": {"label": "normal", "score": 0.1}
    },
    "votes": 0
  }
}
```

### GET /

API status

Response:
```json
{
  "message": "GaitGuard API is running",
  "models_loaded": ["ensemble", "xgboost", "lightgbm", "random_forest", "svm"],
  "anomaly_models_loaded": ["isolation_forest", "lof", "one_class_svm"]
}
```

### GET /health

Health check

Response:
```json
{
  "status": "healthy",
  "models_loaded": 5,
  "anomaly_models_loaded": 3,
  "feature_count": 41,
  "api_version": "2.0.0"
}
```

### GET /docs

Interactive Swagger documentation

Visit: http://localhost:8001/docs

---

## Test Data

Location: test/P1-P6/

6 complete trial datasets with known risk labels:

| Patient | Risk Level | Score | Files |
|---------|-----------|-------|-------|
| P1 | Low | 2-3 | head_raw.csv, lower_back_raw.csv, left_foot_raw.csv, right_foot_raw.csv, metadata.json |
| P2 | Low | 2-3 | Files included |
| P3 | High | 85-88 | Files included (with detected anomalies) |
| P4 | High | 85-88 | Files included |
| P5 | High | 85-88 | Files included |
| P6 | High | 85-88 | Files included |

Each dataset contains:
- 4 sensor CSV files (time, acc_x/y/z, gyr_x/y/z, mag_x/y/z columns)
- 1 metadata.json file
- Complete trial (30-60 seconds of walking)

How to use:
```bash
Test via API
cd api
python start_api.py

Then upload via frontend or curl
curl -X POST http://localhost:8001/predict \
  -F "files=@test/P1/head_raw.csv" \
  -F "files=@test/P1/lower_back_raw.csv" \
  -F "files=@test/P1/left_foot_raw.csv" \
  -F "files=@test/P1/right_foot_raw.csv" \
  -F "files=@test/P1/metadata.json"
```

---

## File Structure

```
GI/
README.md                          This file
datset_link.txt                    Figshare dataset URL
needed_sensors.txt                 Data specification
file_structure                     Directory tree reference

api/
main.py                        FastAPI application (core backend)
start_api.py                   Startup script
requirements.txt               API dependencies
data/
  features/                    Feature cache
  processed/
    signals_clean/             Preprocessed signals

fall_risk_pipeline/
main.py                        Pipeline entry point
requirements.txt               Pipeline dependencies

configs/
pipeline_config.yaml           System configuration

src/
ingestion/
data_loader.py               CSV parsing, metadata validation

preprocessing/
signal_processor.py          Filtering, gait detection, fusion

features/
feature_extractor.py         41 feature computation

models/
trainer.py                   Model training, Optuna tuning
anomaly_detector.py          Unsupervised outlier detection

evaluation/
evaluator.py                 Grouped evaluation, metrics
predictions.py               Generate predictions
reporter.py                  IEEE tables, SHAP plots

data/
README.md                    Data format specification
raw/
  healthy/                   Raw dataset by cohort
  neuro/
  ortho/
processed/
  signals/                   Converted parquet
  signals_clean/             Filtered signals
  trial_metadata.csv         Trial-level metadata
features/
  trial_features.parquet     Per-trial (1K+ rows)
  patient_features.parquet   Aggregated (260 rows)

results/
checkpoints/                 Saved models (.pkl)
  ensemble.pkl
  xgboost.pkl
  lightgbm.pkl
  random_forest.pkl
  svm.pkl

anomaly_detection/           Anomaly models + scalers
  isolation_forest_model.pkl
  lof_model.pkl
  one_class_svm_model.pkl

figures/
  eda/                       Distribution plots, correlations
  models/                    ROC, calibration, confusion matrix
  shap/                      Feature importance visualizations

metrics/
  metrics.csv                AUC, F1, accuracy, etc.
  predictions.csv            Per-sample predictions
  model_comparison_cv.csv   Cross-validation results
  ieee_table.tex             Publication table
  pipeline_report.md         Summary report

Front_end/
index.html                     Main dashboard HTML
main.js                        API client, visualization logic
style.css                      Design system (Apple-inspired)

examples/
head_raw.csv                   Example sensor data
lower_back_raw.csv
left_foot_raw.csv
right_foot_raw.csv
metadata.json

test/
P1/                            Patient 1 (low risk)
  head_raw.csv
  lower_back_raw.csv
  left_foot_raw.csv
  right_foot_raw.csv
  metadata.json
P2/ ... P6/                    5 more test patients
```

---

## Technologies

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI, Uvicorn | High-speed REST service with auto-docs |
| ML | scikit-learn, XGBoost, LightGBM | Classification models |
| Hyperparameter Tuning | Optuna | Bayesian optimization |
| Data Processing | Pandas, NumPy, SciPy | Signal manipulation, feature extraction |
| Explainability | SHAP | Model interpretability (TreeExplainer) |
| Anomaly Detection | scikit-learn | IF, LOF, One-Class SVM |
| Signal Processing | AHRS, scipy.signal | Sensor fusion, filtering, gait events |
| Visualization | Matplotlib, Seaborn, Plotly | Publication-quality plots |
| Logging | Loguru, Rich | Structured logging, pretty output |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Structure | HTML5 | Semantic markup |
| Styling | CSS3 | Apple Design System (dark theme, glassmorphism) |
| 3D Visualization | Three.js | Interactive human model, sensor visualization |
| Signal Plotting | Canvas API, D3.js | Real-time signal visualization |
| API Client | Fetch API, async/await | Async HTTP communication |
| Responsive | CSS Grid, Flexbox | Mobile-friendly layout |

### Utilities

| Tool | Purpose |
|------|---------|
| pytest | Unit testing |
| PyYAML | Configuration management |
| Jinja2 | Report templating |
| pingouin | Statistical tests (DeLong) |
| statsmodels | Calibration metrics |

---

## Model Performance (Historical Internal Results)

| Model | Status | Intended Use | Validation State | Notes |
|-------|--------|--------------|------------------|-------|
| XGBoost | Internal | Baseline | Rerun Required | Grouped validation updated in code |
| LightGBM | Internal | Baseline | Rerun Required | Grouped validation updated in code |
| Random Forest | Internal | Baseline | Rerun Required | Grouped validation updated in code |
| SVM | Internal | Baseline | Rerun Required | Grouped validation updated in code |
| Ensemble | Internal | Research | Rerun Required | Do not cite historical metrics |

### Feature Importance (Top 10 by SHAP)

1. Stride Time Asymmetry (L/R) - 15.2%
2. Head Acceleration RMS - 12.8%
3. Cadence Instability - 11.3%
4. Lateral Asymmetry - 10.9%
5. Lyapunov Exponent (trunk) - 9.8%
6. Spectral Entropy (lower back) - 8.7%
7. Swing Time CV - 7.2%
8. Step Length Asymmetry - 6.8%

---

## Security Features

The application includes the following security measures for production deployment:

- CORS configuration with origin validation
- Environment variable loading via python-dotenv
- File size limits (50MB per file, 200MB total)
- Rate limiting (10 requests per minute)
- File content validation (injection detection, size checks)
- Request validation (file count limits, metadata validation)
- Configurable paths via environment variables
- Health check timeout protection
- Error message control (debug vs production)

---

## Deployment

The application can be deployed to various platforms including:

- Render (recommended for ease of use)
- Railway
- AWS (Elastic Beanstalk, Lightsail)
- DigitalOcean App Platform

See deployment guides for specific platform instructions.

---

## License

This project is intended for research and educational purposes.

---

## Citation

If you use this system in your research, please cite the original dataset:

SpringerNature Figshare DOI: 10.6084/m9.figshare.28806086
