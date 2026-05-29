# Fall-risk label binning

## Problem with legacy binary threshold

`high_risk_threshold: 1` maps **any** non-Healthy cohort to `risk_label = 1`:

| Tier | Cohorts | Reference annual fall rate |
|------|---------|---------------------------|
| 0 — low | Healthy | ~5.2% |
| 1 — moderate (orthopedic) | HipOA, KneeOA, ACL | ~19–29% |
| 2 — high (neurological) | PD, CVA, CIPN, RIL | ~39–67% |

With `label_mode: binary` and `high_risk_threshold: 1`, HipOA (~28.5%) and PD (~67.3%) share the positive class. That conflates orthopedic and neurological pathology and is **not** equivalent to a single clinical risk tier.

## Default: three-class labels

`pipeline_config.yaml` sets `label_mode: multiclass`. Training target `risk_label` equals the pathology tier:

- **0** — Healthy (reference)
- **1** — Orthopedic elevated risk (HipOA, KneeOA, ACL)
- **2** — Neurological elevated risk (PD, CVA, CIPN, RIL)

`multiclass_label` is always stored alongside `risk_label` for audits and sensitivity tables.

Evaluation reports **macro one-vs-rest AUC**, **macro F1**, and **per-class precision/recall/F1** (`per_class_metrics` in metrics payloads).

## Binary modes (optional)

Set `label_mode: binary` only with an explicit clinical rule:

| `binary_strategy` | `high_risk_threshold` | Positive cohorts |
|-------------------|----------------------|------------------|
| `threshold_ge_1` | 1 | All non-Healthy (legacy; heterogeneous) |
| `threshold_ge_2` | 2 | PD, CVA, CIPN, RIL only |

`threshold_ge_2` separates orthopedic from neurological tiers and aligns better with pathology-specific fall-risk literature (Lord SR, Ward JA, Williams P, Anstey KJ. *J Am Geriatr Soc.* 1993;41(11):1226-1234; Allen NE, Schwarzel AK, Canning CG. *Mov Disord.* 2013;28(11):1474-1480).

Sensitivity rows are written to `results/metrics/binary_label_sensitivity.csv` at ingest/report time.

## Literature note

Reference fall probabilities come from the Voisard et al. cohort design (see `COHORT_FALL_PROBABILITIES` in `data_loader.py`). They support **stratified reporting**, not a single cutoff for binarization. A justified binary rule should cite cohort-specific risk thresholds or external calibration — not an arbitrary `≥ 1` on pathology tier alone.

## Configuration

```yaml
dataset:
  label_mode: multiclass          # or binary
  high_risk_threshold: 1          # used when label_mode: binary
  binary_strategy: threshold_ge_1 # or threshold_ge_2
```

Re-run `ingest` (and downstream stages) after changing label policy so `trial_metadata.csv` and feature parquets reflect the new targets.
