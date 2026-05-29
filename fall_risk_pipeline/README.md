# Pathology-Tier Gait Screening Pipeline from Wearable IMU Data

A full professional ML pipeline for research reporting and reproducible internal evaluation, built on the SpringerNature clinical gait dataset (Figshare 28806086).

## Dataset
**A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts**  
- 1,356 trials, 260 participants, 4 IMUs (head, lower back, left/right foot)  
- Cohorts: Healthy | Parkinson's (PD) | CVA | CIPN | RIL | Hip OA | Knee OA | ACL
- Practical sensor-ablation result: **head + right foot (2 sensors) AUC 0.9336**, higher than all 4 sensors (AUC 0.9273)

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

## Pipeline architecture

**Conceptual flow (dependencies, dual tabular + deep paths):** see [`../docs/pipeline_flow.md`](../docs/pipeline_flow.md) — includes a publication-style Mermaid diagram and a code-alignment table.

**Run order:** `main.py` executes the 15 stages below **sequentially** when you run without `--stage`. Partial runs must still respect upstream artifacts (e.g. `train` before `evaluate`).

## Pipeline Stages (15 stages, fully automated)
1. **Ingest** — load raw IMU CSVs, attach metadata via pathologyKey, flag laterality bias  
2. **Validate Gait Events** — compare algorithmic heel-strike detection vs Figshare ground-truth annotations  
3. **Preprocess** — Butterworth filter, Madgwick orientation fusion (head/lower back), gait event segmentation  
4. **EDA** — signal distributions, cohort comparisons, t-SNE, PSD plots  
5. **Features** — temporal, spectral, trunk-dynamics (Lyapunov, ApEn, SampEn, DFA), wavelet, orientation, asymmetry, turning; patient aggregation (mean, std, range, trend)  
6. **Select Features** — RFECV / SHAP pruning to <=20 patient-level features (grouped CV)  
7. **Train** — XGBoost, LightGBM, Random Forest, SVM, MLP with Optuna tuning  
8. **Evaluate** — nested LOSO cross-validation, DeLong/McNemar tests, calibration (Brier + ECE), leakage comparison  
9. **Train Deep** — InceptionTime, Gait Transformer, TCN, CNN-1D, BiLSTM-Attention; **LOSO train + evaluate in one stage** (GPU recommended)  
10. **Ablation** — feature ablation study (all features, top-10 SHAP, leave-one-group-out)  
11. **Sensor Ablation** — which IMU positions are needed? AUC for every sensor subset (1-sensor to 4-sensor)  
12. **Cross-Cohort Transfer** — leave-one-cohort-out: train without PD, test on PD (8x8 pairwise heatmap)  
13. **Predict** — generate out-of-fold predictions for all models  
14. **Anomaly** — unsupervised anomaly detection (Isolation Forest, LOF, One-Class SVM)  
15. **Report** — Sensors-ready tables, figures, demographics, and markdown report  

## Setup
```bash
pip install -r requirements.txt
```

## Reproducible one-command run

From repo root:

```bash
make docker-build
make docker-run
```

Single-stage reproducible run:

```bash
make docker-stage STAGE=evaluate
```

## Model artifacts

Checkpoints and anomaly `.pkl` files are **not** in git. Train locally (`train`, `anomaly` stages) or use `../scripts/download_models.py` with a Hugging Face repo — see `results/checkpoints/README.md` and `../docs/MODEL_CARD.md`.

## Usage
```bash
# Full pipeline end-to-end (all 15 stages)
python main.py --config configs/pipeline_config.yaml

# Individual stages
python main.py --stage ingest
python main.py --stage preprocess
python main.py --stage eda
python main.py --stage features
python main.py --stage select_features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage train_deep
python main.py --stage ablation
python main.py --stage sensor_ablation
python main.py --stage cross_cohort
python main.py --stage predict
python main.py --stage anomaly
python main.py --stage report
```

## Ethics

This pipeline analyzes a publicly available, de-identified dataset (DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)). Original collection was approved by **Comité de Protection des Personnes Île-de-France II** (CPP 2014-10-04 RNI) with written consent (Voisard et al., *Scientific Data* 2025). No new human data were collected. See [`../docs/ethics.md`](../docs/ethics.md).

## Citation
If you use this pipeline, please cite the original dataset:
> Voisard C, et al. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors. *Scientific Data*. https://doi.org/10.1038/s41597-025-05959-w — Figshare https://doi.org/10.6084/m9.figshare.28806086