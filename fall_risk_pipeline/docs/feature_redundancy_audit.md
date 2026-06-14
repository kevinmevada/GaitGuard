# Feature redundancy audit

## Stride-time asymmetry (fixed)

**Issue:** `stride_time_asymmetry` was computed twice in `_asymmetry_features()` — once under that name and again as `stride_time_mean_asymmetry` from the same left/right stride means. Both columns were identical, inflating the feature count without new information.

**Fix:** The duplicate `stride_time_asymmetry` assignment was removed. Bilateral stride asymmetry is reported only as:

- `stride_time_mean_asymmetry`
- `stride_time_std_asymmetry`

**Impact:** Patient-level feature matrices and any models trained on trial features generated **before** this fix must be discarded. Re-run at minimum:

```bash
python main.py --stage features
python main.py --stage select_features   # if selection enabled
python main.py --stage train
```

## Spatial features (step length / gait speed)

**Not computed.** README and config previously listed `step_length_*`, `gait_speed`, and `step_width_*` without implementation. Absolute spatial metrics require calibration or displacement integration not available in the dataset. See `docs/spatial_features.md` for paper wording.

## Spectral centroid (added)

`lb_spectral_centroid` is now computed as \(\sum_k f_k P_k / \sum_k P_k\) on the same Welch PSD as other lower-back spectral features. See `docs/spectral_features.md` for definitions and paper wording.

## Verification

After `features`, confirm `stride_time_asymmetry` is **absent** from `data/features/trial_features.parquet` column list; `stride_time_mean_asymmetry` should be present.

```bash
python -c "import pandas as pd; c=pd.read_parquet('data/features/trial_features.parquet').columns; print('stride_time_asymmetry' in c, 'mean_asym' in str(list(c)))"
```

## Feature-selection re-run (post CRIT-01 / ML-040)

**Context:** RFECV previously ranked features by Gini/MDI importance under p ≫ n, producing a SampEn/DFA-only selected set and depressed classical-model AUC. The fix switches RFECV RFE ranking to **permutation importance** (`rfecv_importance_method: permutation`) and keeps ML-040 required-feature enforcement ranked by mean |SHAP|.

**Status:** Pending full pipeline re-run after the CRIT-01 code change lands. Until regenerated, treat any committed `feature_selection_report.md`, `selected_features.json`, and downstream `metrics.csv` as **stale** relative to current code (see `results/metrics/PIPELINE_VERSION.json` after `report`).

**After re-run, verify:**

1. `selected_features.json` → `rfecv_detail.importance_method` is `"permutation"`.
2. Final 20-feature set is **not** exclusively SampEn/DFA mean/std/range/trend variants.
3. `required_feature_shap_audit.csv` exists when `required_feature_substrings` is set.
4. `feature_selection_report.md` P/N ratios match computed payload values (p/N, not hardcoded literals).
5. Re-run `train` → `evaluate` so classical LOSO AUCs reflect the new mask.

**Paper note (addendum):**

> Feature selection was re-audited after correcting RFECV ranking bias (permutation importance under high p/N). Reported tabular results use feature matrices produced after this re-run; see `feature_redundancy_audit.md` and `PIPELINE_VERSION.json` for provenance.

## Paper note (suggested wording)

> Trial-level features were audited for redundant definitions; a duplicate stride-time asymmetry column (identical to mean-based asymmetry) was removed before final analysis. All reported results use feature matrices produced after this audit.
