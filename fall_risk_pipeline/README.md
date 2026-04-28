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
2. **Preprocessing** — Butterworth filter, Madgwick fusion, gait event segmentation  
3. **EDA** — signal distributions, cohort comparisons, correlation matrices  
4. **Feature Engineering** — temporal, spatial, spectral, variability, asymmetry  
5. **Model Training** — XGBoost, Random Forest, SVM, LightGBM, 1D-CNN, LSTM  
6. **Hyperparameter Tuning** — Optuna-based Bayesian optimization per model  
7. **Model Comparison** — AUC-ROC, F1, calibration plots, and cross-model comparison  
8. **Best Model Selection** — ensemble if beneficial  
9. **Explainability** — SHAP global + local, feature importance  
10. **Report Generation** — IEEE-ready tables and figures  

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Full pipeline end-to-end
python main.py --config configs/pipeline_config.yaml

# Individual stages
python main.py --stage ingest
python main.py --stage preprocess
python main.py --stage eda
python main.py --stage features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage report
```

## Citation
If you use this pipeline, please cite the original dataset:
> Paraschiv-Ionescu et al. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors. *Scientific Data*, Figshare DOI: 10.6084/m9.figshare.28806086