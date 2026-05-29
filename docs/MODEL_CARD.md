# Model card — GaitGuard checkpoints

Hugging Face–style documentation for published `.pkl` artifacts. **Artifacts in git history may be stale**; use a tagged release on the Hub or train from source.

## Model summary

| Field | Value |
|-------|--------|
| **Task** | Participant-level fall-risk classification (multiclass by default) + trial-level unsupervised anomaly screening |
| **Inputs** | Patient-level feature vector (~≤20 selected features after RFECV) from 4-sensor IMU trials |
| **Training data** | Voisard et al. Figshare cohort — 260 participants, 1356 trials (not bundled in repo) |
| **CV** | Leave-one-subject-out, grouped by `participant_id` |
| **Framework** | scikit-learn pipelines, XGBoost, LightGBM |

## Files (Hub layout)

Upload these paths relative to the repository root of the Hub model repo:

### Classification (`checkpoints/`)

| File | Description |
|------|-------------|
| `checkpoints/xgboost.pkl` | Tuned XGBoost pipeline |
| `checkpoints/lightgbm.pkl` | Tuned LightGBM pipeline |
| `checkpoints/random_forest.pkl` | Tuned Random Forest pipeline |
| `checkpoints/svm.pkl` | Tuned RBF SVM pipeline |
| `checkpoints/ensemble.pkl` | Primary ensemble (soft voting or config default) |

Optional: `checkpoints/ensemble_stacking.pkl` if stacking comparison was exported.

### Anomaly (`anomaly_detection/`)

| File | Description |
|------|-------------|
| `anomaly_detection/isolation_forest_model.pkl` | Isolation Forest |
| `anomaly_detection/isolation_forest_scaler.pkl` | StandardScaler for IF |
| `anomaly_detection/lof_model.pkl` | Local Outlier Factor |
| `anomaly_detection/lof_scaler.pkl` | Scaler for LOF |
| `anomaly_detection/one_class_svm_model.pkl` | One-Class SVM |
| `anomaly_detection/one_class_svm_scaler.pkl` | Scaler for OCSVM |
| `anomaly_detection/trial_feature_schema.json` | Trial feature column order for scaler.transform |

### Companion data (same Hub repo or separate dataset repo)

| File | Description |
|------|-------------|
| `data/features/patient_features.parquet` | Feature matrix used for training (regenerate locally preferred) |
| `data/features/selected_features.json` | RFECV/SHAP feature list |
| `data/features/trial_features.parquet` | Trial-level features |

## How to publish to Hugging Face

1. Train with current `main` branch: `python main.py` in `fall_risk_pipeline/`.
2. Create a model repo on [huggingface.co](https://huggingface.co/new) (type: Model).
3. Upload the files above preserving directory structure, or:

```bash
huggingface-cli upload your-org/gaitguard-models \
  fall_risk_pipeline/results/checkpoints/*.pkl checkpoints/ \
  --repo-type model
```

4. Set `GAITGUARD_HF_REPO=your-org/gaitguard-models` for consumers.

## How to download

```bash
pip install huggingface_hub
export GAITGUARD_HF_REPO=your-org/gaitguard-models
python scripts/download_models.py
```

## Ethics

This project uses the public de-identified dataset [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086). Original collection was approved by **Comité de Protection des Personnes Île-de-France II** (CPP 2014-10-04 RNI) with written consent (Voisard et al., *Scientific Data* 2025). No new human data were collected. See `docs/ethics.md` and `docs/paper/ethics_statement.md`.

## Limitations

- **Research prototype — not for clinical use without validation.** Retrospective, single-dataset study; no prospective follow-up or ground-truth fall outcomes (cohort labels only). See `docs/limitations.md`.
- Screening text is intentionally soft (“may warrant further clinical assessment”); not a directive for treatment or intervention.
- Checkpoints are valid only for the **feature schema and label policy** used at train time (see `configs/pipeline_config.yaml`, commit hash in release notes).
- API inference uses **single-trial** projection; see `docs/inference_single_trial_limitation.md`.
- Not a medical device; screening / research use only.

## Training hyperparameters

See `fall_risk_pipeline/configs/pipeline_config.yaml` (`models.tuning`, `models.ensemble`) and `results/metrics/model_comparison_cv.csv` after local training.

## Citation

Voisard et al. (2025). A Dataset of Clinical Gait Signals with Wearable Sensors. Figshare DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086).
