# Feature Selection Report

## Sample size vs dimensionality

- Participants (N): **260**
- Features before selection (p): **464**
- Features exported for training: **20**
- Configured `max_features` cap: **20**
- P/N ratio before (p/N): **1.78**
- P/N ratio after (p/N): **0.08**

Patient-level features aggregate each trial biomarker as mean/std/range/trend, yielding p=464 columns for N=260 participants (P/N ≈ 1.78). With p ≫ N, grouped feature selection is applied before final training to reduce P/N to ≈ 0.08.

## Methods

### RFECV ranking (with optional dimensionality cap)

Recursive Feature Elimination with subject-grouped cross-validation (RFECV), following the RFE framework of Guyon & Elisseeff (2002). The selector uses StratifiedGroupKFold so no participant appears in both train and validation. RFE elimination ranks features by **permutation importance** (not Gini/MDI) to avoid high-variance bias when p >> n.

- RFECV grouped-CV optimal feature count: **45**
- Features exported after cap: **1**

> **Cap applied:** RFECV cross-validation favoured 45 features, but `max_features=20` deliberately limits the export to the top-ranked features (dimensionality cap to lower P/N). Do **not** report this as 'RFECV selected 20 features' — report primary method as **rfecv_capped**.

### SHAP-based pruning (secondary ranking)

Mean absolute SHAP values from a grouped-CV-safe surrogate Random Forest provide an alternate top-20 ranking for comparison. Global rankings average |SHAP| across classes; see `shap_detail.per_class_top_mean_abs_shap` in selected_features.json for tier-specific rankings.

- Primary method used for training: **rfecv_capped**

## Before / after (grouped CV, Random Forest surrogate)

> The **after_global_selection** row uses a feature mask fit on all participants and is **exploratory only** — not a nested selection estimate. Primary LOSO evaluation uses per-fold selection when `nested_in_evaluation: true`.


## Selected features

`f1`

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
- Guyon, I., & Elisseeff, A. (2002). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182.
