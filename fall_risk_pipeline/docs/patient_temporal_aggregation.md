# Patient-level temporal aggregation

Trial-level features are combined per participant using four statistics:

| Suffix | Definition |
|--------|------------|
| `_mean` | Average across trials in the session |
| `_std` | Standard deviation across trials |
| `_range` | max − min across trials (intra-session variability) |
| `_trend` | OLS slope vs trial order (0 … n−1), trials sorted by `session`, then `trial_id` |

**Interpretation:** `_trend` > 0 indicates the feature increases over successive trials (e.g. fatigue-related drift); `_range` captures how much a metric fluctuates within the session without assuming direction.

Configure in `configs/pipeline_config.yaml` under `features.patient_aggregation`.

**Paper note (training, suggested):**  
> Patient-level predictors summarized each trial feature by mean, standard deviation, range, and linear trend across ordered within-session trials, preserving intra-session variability and systematic session effects beyond static averaging.

**Deployment caveat (API inference):**  
The REST API (`POST /predict`) scores **one trial per request**. Trial values are copied into `_mean` columns with `_std = 0`, `_range = 0`, and `_trend = NaN` so the vector matches the trained schema; this is **not** the same as aggregating multiple trials per participant (see `docs/inference_single_trial_limitation.md`). API `confidence` is the model’s maximum class probability, not calibrated clinical certainty.
