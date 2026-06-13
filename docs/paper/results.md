# Results

> **Artifact status:** `fall_risk_pipeline/results/metrics/metrics.csv` was not found at generation time.
>
> **Action required:** run the full post-fix pipeline, then regenerate this file:
>
> ```bash
> cd fall_risk_pipeline
> python main.py
> python ../scripts/regenerate_paper_results.py
> ```
>
> Until regeneration completes, **do not cite numerical results** from older manuscript drafts.
> Prior tables mixed pre-fix code with grouped CV ablation protocols that no longer match the codebase.

## Protocol (current code — post ML-014 fixes)

| Analysis | Validation protocol |
|----------|---------------------|
| Primary tabular LOSO | Leave-one-subject-out + nested RFECV per train fold |
| Primary deep LOSO | Leave-one-subject-out; fixed global DL hyperparams unless `loso_hyperparameter_tuning.enabled` |
| Train `model_comparison_cv.csv` | Nested StratifiedGroupKFold + per-outer-fold RFECV |
| Feature ablation | LOSO (`feature_ablation.py`) |
| Sensor ablation | LOSO (`sensor_ablation.py`) |
| Cross-cohort transfer | LOCO + nested RFECV per train fold |

## Cohort composition (dataset constants)

N = 260 participants, 1,356 trials, eight cohorts, four IMUs — see `docs/paper/methods.md`.
