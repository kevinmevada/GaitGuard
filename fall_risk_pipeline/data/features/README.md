# Feature matrices (generated)

Patient- and trial-level feature parquets and selection artifacts are **gitignored**. This file is tracked in empty clones (STR-003).

## Create features

```bash
cd fall_risk_pipeline
python main.py --stage features
python main.py --stage select_features   # optional RFECV / selected_features.json
```

Inputs: processed signals under `data/processed/`. Outputs: columnar feature files consumed by `train` and `evaluate`.

## Note

The repo root `data/features/README.md` points here. All pipeline paths in `configs/pipeline_config.yaml` are relative to `fall_risk_pipeline/`.
