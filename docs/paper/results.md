# Results

## 1. Cohort composition

The final analysis set contained 260 participants and 1,356 walking trials across eight cohorts (Healthy, HipOA, KneeOA, ACL, PD, CVA, CIPN, RIL) with four synchronized IMUs (head, lower back, left foot, right foot). Demographic details are summarized in `fall_risk_pipeline/results/metrics/table1_demographics.md`.

## 2. Primary multiclass screening performance (tabular models)

Under nested participant-grouped evaluation (`nested_group_cv`), the best overall macro one-vs-rest AUC was achieved by random forest (AUC 0.9164, 95% CI 0.8874-0.9422). Ensemble soft voting and stacking were tied at AUC 0.9113.

| Model | AUC | 95% CI | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|
| random_forest | 0.9164 | [0.8874, 0.9422] | 0.7269 | 0.7846 |
| ensemble_soft_voting | 0.9113 | [0.8801, 0.9382] | 0.7350 | 0.7885 |
| ensemble_stacking | 0.9113 | [0.8801, 0.9382] | 0.7350 | 0.7885 |
| xgboost | 0.9097 | [0.8775, 0.9376] | 0.7133 | 0.7808 |
| lightgbm | 0.8929 | [0.8558, 0.9257] | 0.7212 | 0.7654 |
| svm | 0.8848 | [0.8481, 0.9159] | 0.6841 | 0.7500 |
| mlp | 0.8264 | [0.7834, 0.8700] | 0.6578 | 0.7308 |

## 3. Class-wise behavior

Using per-class one-vs-rest metrics:

- Random forest showed strongest discrimination for the neurological class (class-2 AUC 0.9424; class-2 F1 0.9007).
- Orthopedic class remained the hardest class across models (e.g., random forest class-1 F1 0.6087; ensemble class-1 F1 0.6222).
- Healthy class F1 ranged from 0.6099 (SVM) to 0.6944 (XGBoost); ensemble healthy-class F1 was 0.6892.

## 4. Deep learning LOSO benchmark

Deep models were evaluated with LOSO at participant level (N=260). In this export, deep-model macro-F1 and accuracy are available, while AUC entries are currently empty in `deep_learning_metrics.csv`.

| Deep model | Macro-F1 | Accuracy |
|---|---:|---:|
| dl_tcn | 0.8029 | 0.8346 |
| dl_inception_time | 0.7847 | 0.8231 |
| dl_cnn1d | 0.7771 | 0.8231 |
| dl_bilstm_attention | 0.7720 | 0.8115 |
| dl_gait_transformer | 0.7506 | 0.8038 |

## 5. Explainability (SHAP)

Global random-forest SHAP ranking (`shap_importance_random_forest.csv`) was dominated by lower-back trunk-dynamics and AP-range features:

1. `lb_range_ap_std` (mean |SHAP| = 0.1424)
2. `lb_jerk_max_ml_mean` (0.0670)
3. `lb_std_ml_mean` (0.0644)
4. `lb_rms_ap_mean` (0.0365)
5. `lb_range_ap_mean` (0.0326)

Clinical translation of the top feature:

- `lb_range_ap_std` is the **between-trial variability** (across repeated walks) of the lower-back anterior-posterior acceleration range.
- In plain terms, it captures how consistently a participant controls forward propulsion and braking from one trial to the next.
- Higher values suggest less stable trunk-level gait control (greater walk-to-walk fluctuation), which is clinically compatible with impaired motor control, fatigue effects, or compensatory strategies in higher-risk cohorts.

Cohort-specific patterns from `shap_importance_random_forest_per_cohort.csv` showed the same dominant feature family with cohort-dependent effect size:

- PD: `lb_range_ap_std` (0.1663), `lb_jerk_max_ml_mean` (0.0691), `lb_std_ml_mean` (0.0435)
- CVA: `lb_range_ap_std` (0.2077), `lb_jerk_max_ml_mean` (0.0904), `lb_std_ml_mean` (0.0363)
- Healthy: `lb_range_ap_std` (0.0911), `lb_std_ml_mean` (0.0827), `lb_jerk_max_ml_mean` (0.0532)

## 6. Feature-selection and ablation analyses

Feature-selection comparison (`feature_selection_comparison.csv`) reported:

- Before selection: 392 features, grouped CV AUC 0.9269 +- 0.0125
- After selection: 20 features, grouped CV AUC 0.7951 +- 0.0398

Feature-ablation study (`feature_ablation.md`) using XGBoost LOSO re-fit showed robust performance across feature-group removals:

- All-features scenario: AUC 0.939, macro-F1 0.819
- Top-10 SHAP-only scenario: AUC 0.942, macro-F1 0.808
- Best observed ablation row: minus trunk dynamics, AUC 0.945, macro-F1 0.819

## 7. Sensor ablation

Sensor-subset evaluation (`sensor_ablation.csv`) indicated strong performance with reduced hardware:

- Headline result: head + right foot (2 sensors) achieved AUC 0.9336, which exceeded the full 4-sensor setup (AUC 0.9273)
- Best single sensor: head only (AUC 0.9301)
- Lower-back only remained competitive (AUC 0.9133)
- Foot-only subsets were weaker (left+right foot AUC 0.8552; single foot ~0.843)

Disclosure: these sensor-ablation values come from participant-grouped **5-fold StratifiedGroupKFold** CV (subset-comparison protocol), not full LOSO. They are valid for ranking sensor subsets within the ablation experiment but should not be interpreted as numerically identical to primary LOSO model-performance estimates.

## 8. Cross-cohort transfer

In leave-one-cohort-out transfer (`cross_cohort_transfer.csv`), AUC was undefined for all held-out cohorts because each test cohort contained a single class (`auc_status=undefined_single_class_test`), and one cohort had insufficient known samples. Accordingly, transfer interpretation relied on fallback metrics:

- Highest mean true-class probability: PD (0.7267), CIPN (0.7111)
- Lower transfer confidence: HipOA (0.1999), KneeOA (0.1651), ACL (0.2206)
- Accuracy varied widely (0.0000 to 0.9167), highlighting limited cross-cohort transportability without cohort-specific examples

## 9. Leakage sensitivity check

Grouped-vs-ungrouped comparison (`leakage_comparison.csv`) quantified modest optimism from ungrouped K-fold for most tabular models:

- XGBoost: +0.0059 AUC inflation
- LightGBM: +0.0208 inflation
- SVM: +0.0062 inflation
- Random forest: -0.0025 (no inflation in this run)

Across models, inflation was small overall (mean ~+0.2 percentage points; maximum +2.33 percentage points for LightGBM), and two models showed negative deltas. In this dataset, leakage sensitivity therefore appears limited and is better interpreted as a reassuring robustness check than as a dominant performance driver.

## 10. Prior-work comparison (Table 2)

A structured prior-work comparison is provided in `docs/paper/table2_prior_work.md`, including requested legacy references and comparability notes.

Headline positioning from Table 2:

- This study reports multiclass pathology-tier screening AUC 0.9164 (random forest) on N=260 across eight cohorts.
- Reported 2-sensor performance (head + right foot, AUC 0.9336) is a practical deployment finding not typically reported in older studies.
- Many legacy comparator papers either target different endpoints (for example, fall-event detection) or do not report ROC-AUC, so direct numeric ranking must be interpreted cautiously.
