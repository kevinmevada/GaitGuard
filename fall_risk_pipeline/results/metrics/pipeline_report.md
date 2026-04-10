# Fall Risk Prediction Pipeline - Results Report
Generated: 2026-04-09 23:54

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
| xgboost * | 1.000 | 1.000 | 1.000 | 1.000 |
| lightgbm | 1.000 | 1.000 | 1.000 | 1.000 |
| svm | 1.000 | 1.000 | 1.000 | 1.000 |
| ensemble | 1.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 1.000 | 1.000 | 1.000 | 1.000 |


## Best Model
**xgboost**

- AUC: **1.0000**
- Accuracy: **1.0000**
- F1 Score: **1.0000**
- Sensitivity: **1.0000**

## Outputs
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Reproducibility

python main.py --config configs/pipeline_config.yaml
