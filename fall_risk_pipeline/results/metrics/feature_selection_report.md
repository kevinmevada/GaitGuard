# Feature Selection Report

## Sample size vs dimensionality

- Participants (N): **269**
- Features before selection (p): **2408**
- Features exported for training: **20**
- Configured `max_features` cap: **20**
- P/N ratio before (p/N): **8.95**
- P/N ratio after (p/N): **0.07**

Patient-level features aggregate each trial biomarker as mean/std/range/trend, yielding p=2408 columns for N=269 participants (P/N ≈ 8.95). With p ≫ N, grouped feature selection is applied before final training to reduce P/N to ≈ 0.07.

## Methods

### RFECV ranking (with optional dimensionality cap)

Recursive Feature Elimination with subject-grouped cross-validation (RFECV), following the RFE framework of Guyon & Elisseeff (2002). The selector uses StratifiedGroupKFold so no participant appears in both train and validation. RFE elimination ranks features by **permutation importance** (not Gini/MDI) to avoid high-variance bias when p >> n.

- RFECV grouped-CV optimal feature count: **1688**
- Features exported after cap: **20**

> **Cap applied:** RFECV cross-validation favoured 1688 features, but `max_features=20` deliberately limits the export to the top-ranked features (dimensionality cap to lower P/N). Do **not** report this as 'RFECV selected 20 features' — report primary method as **rfecv_capped**.

### Required nonlinear families (ML-040)

- Forced into final set: **4**
- Dropped by `max_required_features` cap: **16**
- Candidates are ranked by mean |SHAP| before the cap is applied.
- Review `required_feature_shap_audit.csv` after rerun to compare forced vs dropped nonlinear features.
- **Sensitivity (MED-005):** re-run with `required_feature_substrings: []` and compare LOSO AUC / SHAP ranks to this default.

### SHAP-based pruning (secondary ranking)

Mean absolute SHAP values from a grouped-CV-safe surrogate Random Forest provide an alternate top-20 ranking for comparison. Global rankings average |SHAP| across classes; see `shap_detail.per_class_top_mean_abs_shap` in selected_features.json for tier-specific rankings.

- Primary method used for training: **rfecv_capped**

## Before / after (grouped CV, Random Forest surrogate)

> The **after_global_selection** row uses a feature mask fit on all participants and is **exploratory only** — not a nested selection estimate. Primary LOSO evaluation uses per-fold selection when `nested_in_evaluation: true`.

- **before_all_features** (exploratory (global mask)): p=2408, AUC=0.9416 ± 0.0148
- **after_global_selection_grouped_cv_exploratory** (exploratory (global mask)): p=20, AUC=0.8542 ± 0.0192

## Selected features

`head_sampen_mean`, `lb_dfa_trend`, `head_sampen_range`, `lb_dfa_mean`, `it_ms_mp_29_std`, `it_ms_mp_29_mean`, `it_ms_mp_28_trend`, `it_ms_mp_28_range`, `it_ms_mp_28_std`, `it_ms_mp_28_mean`, `mr_f00077_trend`, `mr_f00077_range`, `mr_f00077_std`, `mr_f00077_mean`, `mr_f00076_trend`, `mr_f00076_range`, `mr_f00076_std`, `mr_f00076_mean`, `mr_f00075_trend`, `mr_f00075_range`

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
- Guyon, I., & Elisseeff, A. (2002). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182.
