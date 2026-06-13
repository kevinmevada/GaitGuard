# Anomaly detection artifacts (generated)

Isolation Forest, LOF, and One-Class SVM models plus scalers are **gitignored**. This placeholder is tracked (STR-003).

## Create artifacts

```bash
cd fall_risk_pipeline
python main.py --stage anomaly
```

Outputs include fitted models under this directory and summary metrics in `results/metrics/`.

## Download

Hub layout mirrors training outputs:

```bash
python scripts/download_models.py
```

See `docs/MODEL_CARD.md` for expected filenames.
