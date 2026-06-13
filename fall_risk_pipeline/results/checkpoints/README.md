# Model checkpoints (generated)

Saved `.pkl` pipelines and `checkpoint_manifest.json` are **gitignored**. This file is tracked so the directory exists in a fresh clone (STR-003).

## Create checkpoints

```bash
cd fall_risk_pipeline
python main.py --stage train
```

Ensemble and base models are written here per `configs/pipeline_config.yaml`. The manifest is updated via `src.utils.checkpoint_io.save_checkpoint`.

## Download instead of training

```bash
python scripts/download_models.py
```

Set `GAITGUARD_HF_REPO` (and `GAITGUARD_HF_REVISION` in production) per `docs/MODEL_CARD.md`.

## API / inference

`MODEL_DIR` in `api/.env.example` points here. Production loads checkpoints only after manifest / HMAC verification (SEC-009).
