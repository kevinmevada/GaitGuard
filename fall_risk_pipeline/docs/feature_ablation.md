# Feature ablation study

## Purpose

The main pipeline trains four classifiers plus an ensemble on **≤20 RFECV-selected** patient-level features. Reviewers expect a **feature ablation**: which *groups* (temporal, spectral, trunk dynamics, …) drive macro AUC, and whether compact SHAP subsets match the full matrix.

## Scenarios

| ID | Description |
|----|-------------|
| `all_features` | Full patient-level matrix (all numeric aggregations; **no** `selected_features.json` mask) |
| `top10_shap` | Top 10 columns by mean \|SHAP\| on the full matrix (TreeExplainer on reference model) |
| `minus_<group>` | Leave-one-out for each config group: `temporal`, `spectral`, `trunk_dynamics`, `orientation`, `asymmetry`, `turning` |
| `minus_lyapunov` | Drop only `lyapunov_{mean,std,range,trend}` (isolates nonlinear dynamics vs all of `trunk_dynamics`) |

Trial-level feature count is **33** in the current config (8+7+8+6+2+2). Patient-level columns are **up to 4×** that (mean, std, range, trend), typically **~132** columns before RFECV—not “41”; the ablation table reports actual `n_features` per scenario.

## Method

- **Validation:** leave-one-subject-out (same grouping as evaluation).
- **Model:** `ablation.reference_model` checkpoint (default `xgboost`); hyperparameters from training, **re-fit per fold** on each feature subset (architecture supports variable `p`).
- **Metric:** macro one-vs-rest AUC (multiclass default) + bootstrap 95% CI (`ablation.n_bootstrap`).

## Run

```bash
cd fall_risk_pipeline
python main.py --stage train      # checkpoints required
python main.py --stage ablation
python main.py --stage report     # embeds feature_ablation.md in pipeline_report.md
```

## Outputs

- `results/metrics/feature_ablation.csv`
- `results/metrics/feature_ablation.md`
- `results/metrics/ablation_top_shap_features.json`
- `results/metrics/ablation_group_column_counts.csv`
- `results/figures/models/feature_ablation_bars.pdf`

## Paper subsection

Use `feature_ablation.md` / CSV as **Results — Feature ablation**. Compare `all_features` vs `top10_shap` for parsimony, and each `minus_*` row vs `all_features` for group importance. For Lyapunov specifically, compare `minus_lyapunov` to `minus_trunk_dynamics`.
