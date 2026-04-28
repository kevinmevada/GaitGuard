# Fall Risk Prediction Pipeline - Results Report
Generated: 2026-04-28 01:18

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
| lightgbm * | 0.932 | 0.869 | 0.880 | 0.812 |
| ensemble | 0.920 | 0.869 | 0.881 | 0.818 |
| random_forest | 0.915 | 0.862 | 0.875 | 0.818 |
| xgboost | 0.908 | 0.865 | 0.880 | 0.831 |
| svm | 0.895 | 0.869 | 0.888 | 0.877 |


## Best Model
**lightgbm**

- AUC: **0.9322**
- Accuracy: **0.8692**
- F1 Score: **0.8803**
- Sensitivity: **0.8117**

## Outputs
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Reproducibility

python main.py --config configs/pipeline_config.yaml
