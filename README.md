```md
# GaitGuard: Wearable IMU Gait Screening System

A machine learning pipeline for pathology-tier gait screening from multi-sensor wearable IMU data across eight clinical cohorts.

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

GaitGuard is a research-oriented pathology-tier gait screening pipeline built on open wearable IMU data.
The primary task is supervised 3-class screening (Healthy vs orthopedic vs neurological), with anomaly detection provided as a secondary analysis module.

### Purpose

- Train and evaluate subject-grouped supervised screening models for 3-class pathology tiers
- Quantify performance with grouped CV metrics (AUC, macro-F1, accuracy), calibration, and leakage checks
- Provide explainability via SHAP feature contribution analysis
- Run anomaly detection (Isolation Forest, LOF, One-Class SVM) as complementary unsupervised analysis
- Support reproducible fall-risk research workflows (non-diagnostic)

### Key Statistics

- Status: research prototype with internal evaluation artifacts in this repository
- Dataset: 1,356 trials from 260 participants (8 open-data cohorts)
- Features: trial-level temporal, spectral (incl. centroid), trunk, orientation, and asymmetry indicators (no uncalibrated step length / gait speed)
- Models: tabular ensemble (XGBoost, LightGBM, Random Forest, SVM) + deep models (InceptionTime, Transformer, TCN, CNN-1D, BiLSTM-Attention)
- Sensor ablation: see [`docs/paper/results.md`](docs/paper/results.md) section 7 after `python main.py --stage sensor_ablation` and `python ../scripts/regenerate_paper_results.py` (do not cite pre-fix AUCs from older drafts)
- Anomaly Detection: 3 unsupervised methods (Isolation Forest, LOF, One-Class SVM)

### Open Data Cohorts

- Healthy (baseline gait patterns)
- Hip OA, Knee OA
- ACL injury, CIPN, RIL
- Parkinson's Disease, CVA/Stroke

Data Source: SpringerNature Figshare DOI: 10.6084/m9.figshare.28806086

Input schema reference: [`docs/input_data_specification.md`](docs/input_data_specification.md)

### Ethics

This study used a publicly available, de-identified dataset (DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)). The original data collection was approved by the **Comité de Protection des Personnes Île-de-France II** (CPP 2014-10-04 RNI), with written informed consent obtained by the original investigators (Voisard et al., *Scientific Data* 2025, [10.1038/s41597-025-05959-w](https://doi.org/10.1038/s41597-025-05959-w)). **No new human data were collected.** See [`docs/ethics.md`](docs/ethics.md) and [`docs/paper/ethics_statement.md`](docs/paper/ethics_statement.md).

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
- Feature extraction (trial-level IMU features)
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
[FEATURES] Extract trial-level features per walk, aggregate to patient level
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
[EXTRACT FEATURES] Compute trial-level features from processed signals
[BUILD FEATURE VECTORS]
Patient-level schema for risk model: **single trial** mapped to
`_mean` (trial value), `_std=0`, `_range=0`, `_trend=NaN` — not
multi-trial patient aggregation (training used ~5 trials/participant).
Trial-level vector: median imputation for anomaly models.
See `docs/inference_single_trial_limitation.md`.

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

15-stage ML pipeline. **Architecture diagram (tabular + deep paths, stage dependencies):** [`docs/pipeline_flow.md`](docs/pipeline_flow.md).

| Stage | Module | Purpose |
|-------|--------|---------|
| Ingest | `src/ingestion/data_loader.py` | Parse raw CSV files, validate metadata, create trial records |
| Validate Gait Events | `src/preprocessing/gait_events_gt.py` | Compare algorithmic vs ground-truth heel-strike annotations |
| Preprocess | `src/preprocessing/signal_processor.py` | Butterworth filtering, gravity removal, gait event detection, sensor fusion |
| EDA | `src/visualization/eda.py` | Signal distributions, cohort comparisons, t-SNE, PSD |
| Features | `src/features/feature_extractor.py` | Temporal, spectral, trunk-dynamics, wavelet, orientation features |
| Select Features | `src/features/feature_selector.py` | RFECV / SHAP pruning to <=20 features (grouped CV) |
| Train | `src/models/trainer.py` | Optuna tuning, XGBoost, LightGBM, RF, SVM, MLP |
| Evaluate | `src/evaluation/evaluator.py` | LOSO cross-validation, SHAP, calibration, leakage comparison |
| Train Deep | `src/models/deep_trainer.py` | InceptionTime, Transformer, TCN, CNN-1D, BiLSTM-Attention (LOSO train + eval in one stage) |
| Ablation | `src/evaluation/feature_ablation.py` | Feature ablation study (LOSO AUC per group) |
| Sensor Ablation | `src/evaluation/sensor_ablation.py` | Which IMU positions are needed? (1-to-4 sensor subsets) |
| Cross-Cohort | `src/evaluation/cross_cohort_transfer.py` | Leave-one-cohort-out transfer + pairwise heatmap |
| Predict | `src/evaluation/predictions.py` | Generate out-of-fold predictions for all models |
| Anomaly | `src/models/anomaly_detector.py` | Unsupervised outlier detection (IF, LOF, OC-SVM) |
| Report | `src/evaluation/reporter.py` | Sensors-ready tables, figures, demographics, markdown report |

Config: `configs/pipeline_config.yaml`

### REST API (`api/main.py`)

FastAPI service for real-time inference:

Key functions:
- `parse_uploaded_files()` - Handle ZIP or individual file uploads
- `normalize_sensor_dataframe()` - Column mapping, timestamp handling, imputation
- `preprocess_sensor_data()` - Apply signal processing pipeline
- `extract_trial_features()` - Compute features from one processed trial
- `build_patient_feature_vector()` - Degenerate patient-schema projection (1 trial)
- `predict_risk()` - Ensemble inference on patient-schema vector
- `predict_anomaly()` - 3-method majority voting for outlier detection
- `derive_display_features()` - Extract 5 key biomechanics for UI
- `scale_display_shap()` - Z-score normalization for feature contributions

Endpoints:
- `POST /predict` - Upload trial data to get supervised screening output plus anomaly-analysis fields (informational use)
- `GET /` - API status
- `GET /health` - Health check
- `GET /docs` - Interactive Swagger UI

Runs at: `http://localhost:8001`

### Web Frontend (`Front_end/`)

Research dashboard (served at `GET /app` when the API is running):
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
- Method: Peak detection on inverted foot `acc_z` (75th-percentile height, 0.3 s spacing)
- Validation: `python main.py --stage validate_gait_events` compares detected heel strikes to Figshare `gait_events.csv` or `leftGaitEvents` / `rightGaitEvents` in `*_meta.json` (±50 ms); reports precision/recall in `results/metrics/gait_event_validation_summary.csv`
- Purpose: Identify heel strikes, compute stride/step metrics

Gravity Removal
- Applied to: Lower back accelerometer
- Method: Estimate gravity via low-pass Butterworth filtering (0.1 Hz) and subtract to obtain dynamic acceleration
- Purpose: Isolate dynamic acceleration

Sensor Fusion (Madgwick Algorithm)
- Applied to: Head and lower-back IMUs (`ahrs.filters.Madgwick`)
- Input: Accelerometer + gyroscope (optional magnetometer if enabled in config)
- Output: Quaternion time series → trunk tilt, pitch/roll variability, postural sway velocity features (`lb_*`, `head_*`)
- Purpose: Postural stability features for fall-risk modelling (same path in API inference)

### Feature Engineering (trial-level IMU features)

Trial-level blocks (see `fall_risk_pipeline/src/features/feature_extractor.py`):

1. **Temporal / gait cycle** — stride time, cadence, stance ratio, step count (per foot + aggregates)
2. **Spectral (lower back)** — Welch PSD on gravity-reduced `acc_z`: `lb_dominant_freq`, **`lb_spectral_centroid`** (\(\sum fP/\sum P\)), `lb_spectral_entropy`, band power (`lb_power_0_1hz`, `lb_power_1_3hz`, `lb_power_3_10hz`), `lb_harmonic_ratio` (see `fall_risk_pipeline/docs/spectral_features.md`)
3. **Trunk dynamics** — RMS, jerk, Lyapunov, approximate entropy
4. **Orientation (Madgwick)** — tilt, pitch/roll variability, postural sway velocity
5. **Asymmetry** — stride-time and RMS asymmetry indices

**Not extracted:** absolute step length, gait speed, or step width (uncalibrated IMU; see `fall_risk_pipeline/docs/spatial_features.md`).

Patient-level matrices aggregate each trial feature with **mean, std, range, and trend** across ordered within-session trials. Re-run `features` after spectral or aggregation changes. With **P/N ≈ 1.8** (p/N; ~464 features / 260 participants) on the full matrix, use `select_features` (≤20) before final models.

### Class labels and imbalance

| Multiclass (`COHORT_LABEL_MAP`) | Cohorts | Binary `risk_label` (threshold ≥ 1) |
|---|---|---|
| 0 | Healthy | 0 (low risk) |
| 1 | HipOA, KneeOA, ACL | 1 (high risk) |
| 2 | PD, CVA, CIPN, RIL | 1 (high risk) |

Healthy is ~5.2% reference fall probability but can represent a large share of participants; **class counts are reported explicitly** in `results/metrics/class_distribution_report.md` and `class_distribution_by_cohort.csv` before training.

**Class weights:** Random Forest, LightGBM, and SVM use `class_weight='balanced'`. XGBoost uses `scale_pos_weight = n_negative / n_positive` computed on each training split (not tuned via Optuna).

### Feature Selection (≤20 features before final models)

Pipeline stage: `select_features` (after `features`, before `train`).

| Method | Role |
|--------|------|
| **RFECV** | Primary export — recursive feature elimination with subject-grouped CV (Guyon & Elisseeff, 2002) |
| **SHAP pruning** | Secondary ranking — mean \|SHAP\| from a surrogate Random Forest, top-20 features |
| **Before/after report** | Grouped CV AUC on all features vs selected set (`feature_selection_comparison.csv`) |

Artifacts:

- `data/features/selected_features.json`
- `results/metrics/feature_selection_report.md`
- `results/metrics/feature_selection_comparison.csv`

**References**

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. B*, 58(1), 267–288.
- Guyon, I., & Elisseeff, A. (2002). An introduction to variable and feature selection. *J. Mach. Learn. Res.*, 3, 1157–1182.

### Statistical comparison (Table 1)

After subject-grouped evaluation, paired tests use the **same LOSO out-of-fold predictions** for every model:

| Test | Target | Implementation |
|------|--------|----------------|
| **DeLong** | ROC AUC | Sun & Xu (2014); `auc_pairwise_pvalues.csv` |
| **Bootstrap MWU / Wilcoxon** | AUC (sensitivity) | 1,000 paired bootstrap replicates (`pingouin`) |
| **McNemar** | Accuracy / discrete class labels | `statsmodels.stats.contingency_tables.mcnemar` on aggregated per-fold discordant pairs; `mcnemar_pairwise_pvalues.csv` |

`metrics.csv` and `ieee_table.tex` (LaTeX Table 1) include `p_delong_vs_best` and `p_mcnemar_vs_best` vs the reference model (highest AUC by default).

### Classification threshold (Youden J)

Discrete metrics (accuracy, F1, sensitivity) use **Youden J thresholds fit on each LOSO training fold** and applied to the held-out test fold — not tuned on the same OOF predictions (which would be optimistic).

`metrics_threshold_comparison.csv` also reports **fixed 0.5** and **pooled eval-Youden** (optimistic baseline) with Δ accuracy/F1 vs the train-fold strategy.

### Ensemble & Anomaly Methods

- Ensemble strategy: soft voting across XGBoost, LightGBM, Random Forest, SVM
- Unsupervised anomaly methods: Isolation Forest, LOF, One-Class SVM
- Voting: majority (2/3 methods flag anomaly) -> `Detected`

### Explainability (SHAP)

- **LOSO aggregate** (default): SHAP on every held-out fold (checkpoint refit per fold), concatenated OOF explanations → global mean |SHAP| and summary plot
- **full_checkpoint** (optional): `TreeExplainer` on full-data model with training-set background sample
- Artifacts: `results/figures/shap/`, `results/metrics/shap_importance_<model>.csv`

---

## Setup & Configuration

### One-Command Reproducibility (Docker + Make)

```bash
make docker-build
make docker-run
```

This runs the full 15-stage pipeline in a pinned containerized environment. Host folders are mounted so artifacts persist locally:

- `fall_risk_pipeline/data`
- `fall_risk_pipeline/results`
- `fall_risk_pipeline/logs`

To run a single stage reproducibly:

```bash
make docker-stage STAGE=evaluate
```

### Environment Setup

```bash
# Recommended: one-command setup (Linux/macOS/Git Bash)
bash setup_local.sh

# Windows PowerShell
.\setup_local.ps1
```

Or manually:

```bash
python -m venv .venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
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

## Model checkpoints (not in git)

Classification `.pkl` files and anomaly models under `fall_risk_pipeline/results/` are **generated artifacts** (ignored via `*.pkl`). Clonees must either:

1. **Train:** `cd fall_risk_pipeline && python main.py --stage train` and `--stage anomaly`
2. **Download:** set `GAITGUARD_HF_REPO=your-org/gaitguard-models` and run `python scripts/download_models.py` (see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md))

---

## Running the System

### Option 1: Run full pipeline (recommended)

From the repository root:

```bash
export PYTHONHASHSEED=42   # Linux/macOS
python run_local.py
```

PowerShell: `$env:PYTHONHASHSEED = "42"; python run_local.py`

Or: `make local`

Smoke test with synthetic data:

```bash
python run_local.py --use-local-config --seed-data --trials 6
```

### Option 1b: Run via fall_risk_pipeline/main.py

```bash
cd fall_risk_pipeline
PYTHONHASHSEED=42 python main.py --config configs/pipeline_config.yaml
```

Or use `make pipeline` from the repo root (Makefile sets `PYTHONHASHSEED=42` before Python starts).

### Option 1c: Run with Makefile (local or Docker)

```bash
# Local
make pipeline
make stage STAGE=train_deep

# Reproducible containerized run
make docker-build
make docker-run
```

### Option 2: Run API Server

```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Or from repo root:

```bash
make api
```

### Option 3: Open Frontend

```bash
start http://localhost:8001
```

---

## API Endpoints

### `POST /predict`

Upload **one trial** of IMU data. Risk models expect patient-level feature names but receive a **degenerate single-trial projection** (see `docs/inference_single_trial_limitation.md`).

Response includes:
- `risk_score`, `risk_level` — model-derived screening scores
- `inference_scope` — documents single-trial vs patient-level training (260 participants, 1356 trials)
- `limitations` — confidence is max class probability, not calibrated clinical certainty
- `metadata.inference_note` — short duplicate of the scope note for UIs

**Note:** Legacy field names (`risk_probability`) remain for compatibility.

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
make api
# then open http://localhost:8001/app
```

---

## File Structure

Key directories:

- `api/` - FastAPI inference service
- `fall_risk_pipeline/` - training/evaluation pipeline
- `api/static/` - frontend dashboard (served by FastAPI)
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

**Research prototype — not for clinical use without validation.**

- GaitGuard is for research and informational purposes only.
- Outputs are not medical advice, diagnosis, or treatment guidance.
- Screening notes (e.g. “may warrant further clinical assessment”) are non-directive and must not replace clinician judgment.
- The system does not provide clinically validated diagnostics and must not be used for clinical decision-making without independent prospective validation.
- **Retrospective** analysis on a **single** public dataset; **no prospective follow-up** or **participant-level fall outcomes** (cohort-level labels only). See [`docs/limitations.md`](docs/limitations.md) and [`docs/paper/limitations.md`](docs/paper/limitations.md).

---

## License

This project is released under the **MIT License** for research and educational purposes. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite this repository:

```bibtex
@software{gaitguard2026,
  author       = {Mevada, Kevin},
  title        = {GaitGuard: Wearable IMU Gait Screening System},
  year         = {2026},
  url          = {https://github.com/<org-or-user>/GI},
  note         = {Version/commit used in your study}
}
```

Also cite the source dataset:

> Voisard C, et al. A dataset of clinical gait signals with wearable sensors from healthy, neurological, and orthopedic cohorts. *Scientific Data*. 2025. https://doi.org/10.1038/s41597-025-05959-w  
> Dataset: https://doi.org/10.6084/m9.figshare.28806086
