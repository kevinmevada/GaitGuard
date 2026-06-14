# Pipeline results (generated)

This directory holds **generated** metrics, figures, checkpoints, and reports. Contents are **gitignored** except these placeholder READMEs (STR-003).

## Layout

| Path | Produced by |
|------|-------------|
| `metrics/` | `evaluate`, `train_deep`, `ablation`, `sensor_ablation`, `cross_cohort`, `predict`, `anomaly`, `report` (see `metrics/README.md`) |
| `figures/eda/` | `eda` |
| `figures/models/` | `evaluate`, `train_deep` |
| `figures/shap/` | `evaluate` |
| `checkpoints/` | `train` (see `checkpoints/README.md`) |
| `anomaly_detection/` | `anomaly` (see `anomaly_detection/README.md`) |

## Populate locally

From `fall_risk_pipeline/`:

```bash
python main.py                    # full pipeline
python main.py --stage train      # checkpoints only
python main.py --stage evaluate   # primary metrics + figures
```

Or download published artifacts:

```bash
python ../scripts/download_models.py
```

**Before citing metrics in a manuscript**, verify `results/metrics/PIPELINE_VERSION.json` matches your git commit and `configs/pipeline_config.yaml` (written at the start of `report`).

API inference expects checkpoints under `results/checkpoints/` (see `api/.env.example`).
