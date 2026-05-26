# Fall Risk Prediction System from Wearable IMU Gait Data

A full professional ML pipeline for research reporting and reproducible internal evaluation, built on the SpringerNature clinical gait dataset (Figshare 28806086).

## Dataset
**A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts**  
- 1,356 trials, 260 participants, 4 IMUs (head, lower back, left/right foot)  
- Cohorts: Healthy | Parkinson's (PD) | CVA | CIPN | RIL | Hip OA | Knee OA | ACL

## Project Structure
```
fall_risk_pipeline/
├── data/
│   ├── raw/                    # Place downloaded dataset here
│   ├── processed/              # Cleaned & segmented signals
│   └── features/               # Engineered feature matrices
├── src/
│   ├── ingestion/              # Data loading & parsing
│   ├── preprocessing/          # Filtering, segmentation, fusion
│   ├── features/               # Feature extraction (temporal/spectral/spatial)
│   ├── models/                 # All model architectures
│   ├── evaluation/             # Metrics, calibration, SHAP
│   └── visualization/          # EDA and result plots
├── configs/                    # YAML config files
├── notebooks/                  # Exploratory notebooks
├── results/
│   ├── figures/eda/            # EDA plots
│   ├── figures/models/         # ROC, calibration curves
│   ├── figures/shap/           # SHAP plots
│   ├── metrics/                # JSON/CSV metric reports
│   └── checkpoints/            # Saved model weights
├── logs/                       # Training logs
└── tests/                      # Unit tests
```

## Pipeline Stages
1. **Ingestion** — load raw IMU CSVs, attach metadata, flag laterality bias  
2. **Preprocessing** — Butterworth filter, Madgwick fusion (head/lower back), gait event segmentation; optional `validate_gait_events` vs Figshare HS annotations  
3. **EDA** — signal distributions, cohort comparisons, correlation matrices  
4. **Feature Engineering** — temporal gait-cycle + spectral/trunk/orientation/asymmetry features (`docs/spectral_features.md`, `docs/spatial_features.md` for what is **not** extracted); patient aggregation = **mean, std, range, trend**; see `docs/feature_redundancy_audit.md`  
5. **Feature Selection** — RFECV / SHAP pruning to ≤20 patient-level features (grouped CV; before/after report)  
6. **Model Training** — XGBoost, Random Forest, SVM, LightGBM, 1D-CNN, LSTM  
7. **Hyperparameter Tuning** — Optuna-based Bayesian optimization per model  
8. **Model Comparison** — AUC-ROC, F1, calibration plots, and cross-model comparison  
9. **Best Model Selection** — compare soft voting vs logistic stacking (nested LOSO; `ensemble_comparison.csv`)  
10. **Feature ablation** — LOSO AUC for all features, top-10 SHAP, and leave-one-group-out (`docs/feature_ablation.md`)  
11. **Explainability** — SHAP global + local, feature importance  
12. **Report Generation** — IEEE-ready tables and figures  

## Setup
```bash
pip install -r requirements.txt
```

## Model artifacts

Checkpoints and anomaly `.pkl` files are **not** in git. Train locally (`train`, `anomaly` stages) or use `../scripts/download_models.py` with a Hugging Face repo — see `results/checkpoints/README.md` and `../docs/MODEL_CARD.md`.

## Usage
```bash
# Full pipeline end-to-end
python main.py --config configs/pipeline_config.yaml

# Individual stages
python main.py --stage ingest
python main.py --stage preprocess
python main.py --stage eda
python main.py --stage features
python main.py --stage select_features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage ablation
python main.py --stage report
```

## Ethics

This pipeline analyzes a publicly available, de-identified dataset (DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)). Original collection was approved by **Comité de Protection des Personnes Île-de-France II** (CPP 2014-10-04 RNI) with written consent (Voisard et al., *Scientific Data* 2025). No new human data were collected. See [`../docs/ethics.md`](../docs/ethics.md).

## Citation
If you use this pipeline, please cite the original dataset:
> Voisard C, et al. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors. *Scientific Data*. https://doi.org/10.1038/s41597-025-05959-w — Figshare https://doi.org/10.6084/m9.figshare.28806086