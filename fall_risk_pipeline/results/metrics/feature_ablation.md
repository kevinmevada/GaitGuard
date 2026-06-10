# Feature ablation (LOSO macro-OVR AUC)

Reference classifier: **xgboost** (checkpoint hyperparameters, re-fit per LOSO fold).

Trial-level features in config: **52**; patient-level columns vary by aggregation (mean, std, range, trend).

Top-10 SHAP features (full matrix): `head_jerk_max_v_mean`, `lb_jerk_mean_ap_mean`, `head_jerk_mean_ap_mean`, `lb_range_ap_mean`, `lb_wavelet_entropy_mean`, `lb_range_ap_range`, `right_step_count_std`, `lb_spectral_entropy_std`, `lb_jerk_mean_ml_mean`, `head_jerk_mean_ml_mean`

| Scenario | n features | AUC | 95% CI | Macro F1 |
|---|---:|---:|---|---:|
| minus_temporal | 396 | 0.950 | [0.926, 0.969] | 0.791 |
| minus_trunk_dynamics | 416 | 0.946 | [0.920, 0.966] | 0.815 |
| all_features | 464 | 0.946 | [0.920, 0.966] | 0.813 |
| minus_orientation | 416 | 0.946 | [0.921, 0.965] | 0.801 |
| minus_lyapunov | 456 | 0.946 | [0.919, 0.965] | 0.794 |
| minus_asymmetry | 452 | 0.945 | [0.920, 0.966] | 0.782 |
| minus_turning | 448 | 0.944 | [0.918, 0.965] | 0.811 |
| top10_shap | 10 | 0.944 | [0.916, 0.966] | 0.790 |
| minus_spectral | 328 | 0.943 | [0.916, 0.964] | 0.814 |

## Interpretation

- Compare `all_features` vs `top10_shap`: if AUC is similar, a compact SHAP subset may suffice.
- Compare each `minus_*` row to `all_features`: larger AUC drops indicate groups that contribute most.
- `minus_lyapunov` isolates the Lyapunov exponent (under `trunk_dynamics`); compare to `minus_trunk_dynamics`.

Outputs: `feature_ablation.csv`, `ablation_group_column_counts.csv`, `figures/models/feature_ablation_bars.*`