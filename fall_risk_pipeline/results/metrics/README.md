# Metrics artifacts (generated)

CSV/JSON metrics, ablation tables, and provenance files are **gitignored**. This placeholder is tracked (STR-003).

## Provenance

After a full pipeline run, check `PIPELINE_VERSION.json` (written at the start of the `report` stage). It records:

- Git commit (+ dirty flag)
- SHA-256 of `pipeline_config.yaml` / loaded config
- Primary endpoint, feature-selection settings, pipeline seed

Compare this file against your checkout before citing `metrics.csv`, `model_comparison_cv.csv`, or `primary_endpoint.json` in a manuscript.

## Create artifacts

```bash
cd fall_risk_pipeline
export PYTHONHASHSEED=42   # match reproducibility.seed
python main.py --stage select_features --stage train --stage evaluate --stage anomaly --stage report
```

Key outputs:

| File | Stage |
|------|-------|
| `feature_selection_report.md` | `select_features` |
| `required_feature_shap_audit.csv` | `select_features` (when ML-040 substrings set) |
| `nonlinear_nan_report.csv` | `features` (trial-level SampEn/DFA/Lyapunov NaN rates) |
| `metrics.csv`, `model_comparison_cv.csv` | `evaluate` |
| `primary_endpoint.json` | `evaluate` / `anomaly` (depends on `primary_endpoint` config) |
| `PIPELINE_VERSION.json` | `report` |

## Download

Published bundles should ship with a matching `PIPELINE_VERSION.json`:

```bash
python scripts/download_models.py
```
