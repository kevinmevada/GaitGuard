# Single-trial API inference vs patient-level training

## Cohort scale

- **260 participants** (patient-level cross-validation groups by `participant_id`)
- **1356 trials** (~5.2 trials per participant on average)

Training and internal validation use **one feature row per participant**, built by aggregating all trials for that person within a session.

## Training aggregation (offline pipeline)

For each numeric trial feature, the pipeline computes four patient-level statistics across ordered trials (`session`, `trial_id`):

| Suffix | Meaning |
|--------|---------|
| `_mean` | Average across trials |
| `_std` | Standard deviation across trials |
| `_range` | max − min (intra-session variability) |
| `_trend` | Linear slope vs trial order |

See `fall_risk_pipeline/docs/patient_temporal_aggregation.md`.

## API inference (online)

`POST /predict` accepts **one walking trial** per request. `build_patient_feature_vector()` maps that trial into the **same column names** the fitted models expect:

- `{feature}_mean` ← trial value
- `{feature}_std` ← `0.0` (no cross-trial dispersion)
- `{feature}_range` ← `0.0` (no within-session spread)
- `{feature}_trend` ← `NaN` (no trial-order trajectory)

So a single trial is **not** equivalent to a patient-level aggregate from multiple trials; it is a **degenerate** projection onto the training schema.

## Confidence limitation

The API `confidence` field is **`max(predict_proba)`** from the classifier — a model score, not a clinically calibrated probability. With one trial:

- Variability and fatigue/trend signals the model may have learned from `_std`, `_range`, and `_trend` are absent or fixed.
- Scores should be interpreted as **screening-oriented**, not diagnostic.

## Suggested paper wording (Methods or Limitations)

> **Training.** Supervised models were fit on patient-level feature vectors (N = 260). Each predictor summarized trial-level IMU features by mean, standard deviation, range, and linear trend across within-session trials (1,356 trials total; LOSO grouped by participant).
>
> **Deployment inference.** The public API accepts a **single uploaded trial** per request. To match the trained feature schema, that trial’s values populate the patient-level `_mean` columns while `_std` and `_range` are set to zero and `_trend` is undefined. Real-time scores therefore **do not** replicate full patient-level aggregation and may underestimate uncertainty relative to multi-trial participants in the training cohort. We report API outputs as screening aids; the `confidence` field reflects the model’s maximum class probability, not an externally validated clinical certainty.

## API response fields

Each `/predict` response includes `inference_scope` and `limitations` documenting this gap explicitly (see `api/main.py`).

## `display_gauges` (UI sidebar bars)

The JSON field `display_gauges` is marked **`display_only: true`** and **`not_shap: true`**. It powers frontend gauge bars only — **not** SHAP attributions and **not** publishable as model confidence unless sourced as below:

| Bar key | Source |
|---------|--------|
| `confidence` | Classifier `max(predict_proba) × 100` |
| `anomaly` | Share of anomaly detectors voting anomaly |
| `risk_score` | Classifier `risk_score` (0–100) |
| `trials` | `n_trials_in_request` vs mean training trials per participant (~5.2); coverage indicator only |

The legacy `graph_values` object mirrors `display_gauges.values` (with deprecated `cohort` alias = `risk_score`). Do not cite hardcoded cohort maps or fixed anomaly/trial percentages from older API versions in papers.
