# Feature Selection Report

## Sample size vs dimensionality

- Participants (N): **260**
- Features before selection (p): **464**
- Features after selection: **20** (cap ≤ 20)
- P/N ratio before: **0.56**
- P/N ratio after: **13.00**

An ensemble of four nonlinear models on high-dimensional patient-level features (mean/std/range/trend) with N=260 is severely underpowered (P/N ≈ 3.25). We therefore apply grouped feature selection before final training.

## Methods

### RFECV (primary export)

Recursive Feature Elimination with subject-grouped cross-validation (RFECV), following the RFE framework of Guyon & Elisseeff (2002). The selector uses StratifiedGroupKFold so no participant appears in both train and validation.

- RFECV CV-optimal feature count: 297
- Exported RFECV features: 20

### SHAP-based pruning (secondary ranking)

Mean absolute SHAP values from a grouped-CV-safe surrogate Random Forest provide an alternate top-20 ranking for comparison.

- Primary method used for training: **rfecv**

## Before / after (grouped CV, Random Forest surrogate)

- **before_all_features**: p=464, AUC=0.9355 ± 0.0121
- **after_selected_features**: p=20, AUC=0.7951 ± 0.0398

## Selected features

`lb_sampen_mean`, `lb_sampen_std`, `lb_sampen_range`, `lb_sampen_trend`, `lb_dfa_mean`, `lb_dfa_std`, `lb_dfa_range`, `lb_dfa_trend`, `head_sampen_mean`, `head_sampen_std`, `head_sampen_range`, `head_sampen_trend`, `head_dfa_mean`, `head_dfa_std`, `head_dfa_range`, `head_dfa_trend`, `head_lb_dfa_ratio_mean`, `head_lb_dfa_ratio_std`, `head_lb_dfa_ratio_range`, `head_lb_dfa_ratio_trend`

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288.
- Guyon, I., & Elisseeff, A. (2002). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182.
