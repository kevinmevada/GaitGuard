# Clinical thresholds and risk levels

## Problem

Fixed API bands (**risk score ≥ 70 → high**, **≥ 40 → moderate**) were arbitrary: they were not derived from ROC analysis or from established fall-risk screening tools.

## Primary cutoff: Youden J (data-driven)

After `python main.py --stage evaluate`, the pipeline writes:

`fall_risk_pipeline/results/metrics/clinical_threshold.json`

| Field | Meaning |
|-------|---------|
| `primary_cutoff.probability` | **Deployment threshold** — mean LOSO train-fold Youden J on elevated-risk probability |
| `primary_cutoff.sensitivity` | Sensitivity (TPR) at that threshold on pooled OOF predictions |
| `primary_cutoff.specificity` | Specificity (TNR) at that threshold |
| `eval_youden_cutoff` | Youden fit on pooled OOF (slightly optimistic; for comparison only) |
| `fixed_cutoff_0_5` | Metrics at 0.5 for baseline comparison |

**Elevated risk probability**

- **Binary** models: positive-class `predict_proba`.
- **Multiclass** models: sum of class probabilities for tiers ≥ `dataset.high_risk_threshold` (default: tiers 1+2).

## Clinical screening context (citations only)

These tools inform the *clinical problem*; they are **not** used to set the IMU model cutoff.

### Morse Fall Scale (MFS)

- Morse JM, Morse RM, Tylko SJ. *Can J Aging.* 1989;8(4):373-385.
- Common inpatient rule: **score ≥ 45 → high fall risk**.

### STRATIFY

- Oliver D et al. *BMJ.* 1997;315(7115):1049-1053.
- **Score ≥ 5 → high fall risk**.

A calibration study would be required to map MFS/STRATIFY scores to IMU-derived probabilities.

## API risk levels (after fix)

| Level | Rule |
|-------|------|
| **high** | `elevated_probability ≥ Youden J cutoff` |
| **moderate** | `0.5 × Youden ≤ elevated_probability < Youden` (borderline; not a guideline) |
| **low** | `elevated_probability < 0.5 × Youden` |

`/predict` returns `clinical_threshold` with validated sensitivity/specificity and Morse/STRATIFY references.

## Paper wording (Methods / Results)

**Note:** After running the evaluate stage, populate values below from `results/metrics/clinical_threshold.json`.

> Classification used a subject-grouped LOSO protocol. The operating point was chosen by **maximizing Youden’s J** (sensitivity + specificity − 1) on training folds and applying the mean fold threshold to held-out subjects. At this cutoff, the [reference model] achieved sensitivity **X.XX** and specificity **X.XX** on out-of-fold predictions. For clinical context we cite the Morse Fall Scale and STRATIFY inpatient screens; our wearable IMU probability was not calibrated to those instruments.

Replace **X.XX** from `clinical_threshold.json` after evaluation.
