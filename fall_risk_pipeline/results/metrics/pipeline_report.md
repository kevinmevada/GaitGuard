# Fall Risk Prediction Pipeline - Results Report
Generated: 2026-04-11 20:00

## Dataset
- Participants: 260
- Sensors: 4 IMUs
- Label mode: binary

## Validation
- Strategy: nested_group_cv
- Note: Reported metrics are intended for subject-grouped evaluation output, not in-sample prediction export.

## Model Performance

| Model | AUC | Accuracy | F1 | Sensitivity |
|---|---|---|---|---|
| ensemble * | 0.937 | 0.877 | 0.897 | 0.909 |
| lightgbm | 0.933 | 0.850 | 0.872 | 0.864 |
| xgboost | 0.931 | 0.838 | 0.871 | 0.922 |
| random_forest | 0.928 | 0.850 | 0.872 | 0.864 |
| svm | 0.917 | 0.881 | 0.898 | 0.890 |


## Best Model
**ensemble**

- AUC: **0.9370**
- Accuracy: **0.8769**
- F1 Score: **0.8974**
- Sensitivity: **0.9091**

## Outputs
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Reproducibility

python main.py --config configs/pipeline_config.yaml
