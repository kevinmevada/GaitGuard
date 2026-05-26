# Feature matrices (generated — not in git)

Parquet files in this directory are **pipeline outputs**, not source artifacts. They are excluded via `*.parquet` in the repository `.gitignore`.

## Generate locally

From `fall_risk_pipeline/` with raw data under `data/raw/`:

```bash
pip install -r requirements.txt
python main.py --stage ingest
python main.py --stage preprocess
python main.py --stage features
```

Optional downstream stages:

```bash
python main.py --stage select_features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage report
```

Or run the full pipeline:

```bash
python main.py --config configs/pipeline_config.yaml
```

## Expected outputs

| File | Description |
|------|-------------|
| `trial_features.parquet` | One row per walking trial |
| `patient_features.parquet` | Participant-level aggregation (mean, std, range, trend) |
| `selected_features.json` | After `select_features` (if enabled) |

Cloning the repo does **not** include these files; regenerate them so features match your code version and config.
