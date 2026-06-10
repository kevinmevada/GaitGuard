# Pathology-Tier Gait Screening Pipeline - Results Report
Generated: 2026-06-04 01:44

## Dataset
- Participants: 260
- Sensors: 4 IMUs
- Label mode: multiclass — Primary target: 3-class labels (0=Healthy, 1=orthopedic, 2=neurological). Preserves separation between OA/ACL and neurodegenerative/vascular cohorts.

# Table 1 — Demographics by cohort

Participant-level summary from `trial_metadata.csv` (one row per `participant_id`; trials counted separately).

| Cohort | *n* (participants) | Trials | Age (years) | Sex (F/M) | Laterality |
|---|---:|---:|---|---|---|
| Healthy | 73 | 360 | 39.8 ± 20.2 | 32 F / 41 M (44% F) | Right (68); Left (5) |
| HipOA | 15 | 74 | 69.7 ± 13.2 | 8 F / 7 M (53% F) | Right (15) |
| KneeOA | 18 | 78 | 69.7 ± 13.7 | 11 F / 7 M (61% F) | Right (17); Left (1) |
| ACL | 11 | 60 | 36.7 ± 11.9 | 4 F / 7 M (36% F) | Right (9); Left (2) |
| PD | 24 | 160 | 74.2 ± 11.4 | 7 F / 17 M (29% F) | Right (20) |
| CVA | 49 | 128 | 59.0 ± 9.5 | 12 F / 37 M (24% F) | Right (47); Left (2) |
| CIPN | 19 | 98 | 63.4 ± 10.5 | 9 F / 10 M (47% F) | Right (16); Ambidextrous (1) |
| RIL | 51 | 398 | 59.8 ± 13.8 | 23 F / 28 M (45% F) | Right (41); Left (4); Ambidextrous (1) |
| Total | 260 | 1356 | 55.9 ± 19.2 | 106 F / 154 M (41% F) | Right (233); Left (14); Ambidextrous (2) |

Age: mean ± SD among participants with recorded age. Sex: counts and female proportion among participants with recorded sex. Laterality: affected/dominant side when present in metadata.


## Class distribution (training labels)

Label policy: multiclass (see `docs/label_binning.md`). Multiclass tiers: 0=Healthy, 1=orthopedic (HipOA/KneeOA/ACL), 2=neurological (PD/CVA/CIPN/RIL). Legacy binary at threshold≥1 conflates orthopedic and neurological risk.

| Cohort | N | Multiclass | Train label | Binary≥1 | Binary≥2 | Fall % |
|---|---:|---:|---:|---:|---:|---:|
| Healthy | 73 | 0 | 0 | 0 | 0 | 5.2 |
| RIL | 51 | 2 | 2 | 1 | 1 | 38.9 |
| CVA | 49 | 2 | 2 | 1 | 1 | 54.2 |
| PD | 24 | 2 | 2 | 1 | 1 | 67.3 |
| CIPN | 19 | 2 | 2 | 1 | 1 | 41.8 |
| KneeOA | 18 | 1 | 1 | 1 | 0 | 24.1 |
| HipOA | 15 | 1 | 1 | 1 | 0 | 28.5 |
| ACL | 11 | 1 | 1 | 1 | 0 | 18.7 |

See `class_distribution_report.md` for full counts.

## Feature selection (dimensionality control)

With patient-level features (mean, std, range, trend per trial feature) the dimensionality may be high relative to sample size. RFECV (Guyon & Elisseeff, 2002) and SHAP pruning reduce p to ≤20 before final training; see Tibshirani (1996) for Lasso-style sparsity.

| Stage | Features (p) | Grouped CV AUC |
|---|---:|---:|
| before_all_features | 464 | 0.9355 ± 0.0121 |
| after_selected_features | 20 | 0.7951 ± 0.0398 |

Full report: `feature_selection_report.md`

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

## Clinical threshold (Youden J)

Risk classification uses a **data-driven Youden J cutoff** on LOSO out-of-fold elevated-risk probability — not fixed API score bands (70/40).

- **Primary cutoff (deployment):** P(elevated) ≥ **0.712** (risk score ≥ 71)
- **Sensitivity** at primary cutoff: **0.533**
- **Specificity** at primary cutoff: **0.787**
- **PPV / NPV:** 0.892 / 0.591
- Source: `loso_oof_eval_youden` on model `svm`

### Clinical screening context (not used as IMU cutoffs)

- **Morse Fall Scale:** Score ≥ 45 indicates high fall risk (inpatient nursing). (Morse JM, Morse RM, Tinzoh J. Development of a scale to identify the fall-prone patient. Can J Aging. 1989;8(4):373-385.)
- **STRATIFY:** Score ≥ 5 indicates high fall risk. (Oliver D, Britton M, Seed P, Martin FC, Hopper AH. Development and evaluation of evidence based risk assessment tool (STRATIFY) to predict which elderly inpatients will fall. BMJ. 1997;315(7115):1049-1053.)

Full artifact: `clinical_threshold.json`. See `docs/clinical_thresholds.md`.

## Ethics

This study used a publicly available, de-identified dataset (DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)). The original data collection was approved by the **Comité de Protection des Personnes Île-de-France II** (CPP 2014-10-04 RNI), with written informed consent obtained by the original investigators (Voisard et al., *Scientific Data* 2025, [10.1038/s41597-025-05959-w](https://doi.org/10.1038/s41597-025-05959-w)). **No new human data were collected.**

Manuscript text: `ethics_statement.md` (repo `docs/paper/`).


# Prediction export — limitations

**Research prototype — not for clinical use without validation.**

- Research prototype — not for clinical use without validation.
- Retrospective secondary analysis of existing recordings — no forward-in-time data collection for this project.
- Single open dataset (Voisard et al., Figshare 10.6084/m9.figshare.28806086); no external multi-site replication reported.
- No prospective participant follow-up or incident-outcome ascertainment for this analysis.
- No participant-level ground-truth fall outcomes; labels are cohort-level pathology tiers and literature-based fall-risk references only.
- Internal LOSO metrics on the same dataset — not independent prospective performance.
- Outputs are exploratory screening scores from open IMU gait data, not medical advice or a diagnosis.
- Single-trial API inference does not replicate multi-trial patient aggregation used in training.
- Risk thresholds (Youden J) are derived from internal cross-validation, not regulatory clearance.
- Do not use these results for treatment or fall-prevention decisions without prospective validation.


## Ensemble method comparison (nested LOSO)

Base learners: top-K models by grouped CV AUC. **Soft voting** averages positive-class probabilities; **stacking** fits a logistic regression meta-learner on out-of-fold base probabilities (inner StratifiedGroupKFold).

| Method | AUC | AUC 95% CI | F1 |
|---|---:|---|---:|
| stacking | 0.7959 | [0.750, 0.841] | 0.5816 |
| soft_voting | 0.7938 | [0.747, 0.839] | 0.5727 |

## Deep learning comparison (LOSO, raw IMU windows)

Architectures trained end-to-end on windowed raw sensor signals (4 IMUs × 13 channels, 256-sample windows at 100 Hz). Each model evaluated under the same Leave-One-Subject-Out protocol as the classical ML models (N participants).

| Architecture | AUC | 95% CI | Macro F1 | Accuracy |
|---|---:|---|---:|---:|
| Bilstm Attention | 0.9292 | [0.904, 0.952] | 0.7724 | 0.8115 |
| Gait Transformer | 0.9211 | [0.895, 0.945] | 0.7160 | 0.7654 |
| Cnn1D | 0.9182 | [0.890, 0.944] | 0.7414 | 0.7923 |
| Inception Time | 0.9171 | [0.889, 0.943] | 0.7234 | 0.7731 |
| Tcn | 0.9152 | [0.886, 0.942] | 0.7767 | 0.8154 |

Best classical ML: **svm** (AUC 0.7966)
Best deep learning: **dl_bilstm_attention** (AUC 0.9292)

## Sensor Position Ablation

AUC for each subset of the four IMU positions (head, lower back, left foot, right foot). Identifies the minimum sensor configuration for acceptable screening performance.

| Sensor Subset | # Sensors | # Features | AUC (mean) | AUC (std) |
|---|---:|---:|---:|---:|
| head+lowerback | 2 | 380 | 0.9446 | 0.0136 |
| head+lowerback+rightfoot | 3 | 408 | 0.9445 | 0.0141 |
| head+rightfoot | 2 | 204 | 0.9436 | 0.0156 |
| head+lowerback+leftfoot+rightfoot | 4 | 464 | 0.9426 | 0.0150 |
| head+lowerback+leftfoot | 3 | 408 | 0.9420 | 0.0119 |
| head | 1 | 176 | 0.9420 | 0.0195 |
| head+leftfoot+rightfoot | 3 | 260 | 0.9401 | 0.0113 |
| head+leftfoot | 2 | 204 | 0.9393 | 0.0117 |
| lowerback+rightfoot | 2 | 220 | 0.9219 | 0.0249 |
| lowerback | 1 | 192 | 0.9200 | 0.0239 |
| lowerback+leftfoot+rightfoot | 3 | 276 | 0.9193 | 0.0252 |
| lowerback+leftfoot | 2 | 220 | 0.9179 | 0.0282 |
| leftfoot+rightfoot | 2 | 84 | 0.8694 | 0.0382 |
| rightfoot | 1 | 28 | 0.8561 | 0.0418 |
| leftfoot | 1 | 28 | 0.8453 | 0.0284 |

**Best overall subset:** head+lowerback (AUC 0.9446).
**Best single-sensor:** head (AUC 0.9420).
## Cross-Cohort Transfer (Leave-One-Cohort-Out)

Train on all subjects from N-1 cohorts, test on the held-out cohort. Answers: 'Can a model trained without any PD patients still detect PD?'

| Held-Out Cohort | N (test) | AUC | Mean True-Class Prob. | Accuracy | F1 (macro) |
|---|---:|---:|---:|---:|---:|
| ACL | 11 | N/A | 0.1861 | 0.0000 | 0.0000 |
| CIPN | 19 | N/A | 0.7110 | 0.8947 | 0.4722 |
| CVA | 49 | N/A | 0.3837 | 0.4082 | 0.1932 |
| Healthy | 73 | N/A | N/A | nan | nan |
| HipOA | 15 | N/A | 0.1658 | 0.0667 | 0.0417 |
| KneeOA | 18 | N/A | 0.1730 | 0.0000 | 0.0000 |
| PD | 24 | N/A | 0.7281 | 0.9583 | 0.4894 |
| RIL | 51 | N/A | 0.5521 | 0.6078 | 0.2520 |

AUC is **undefined** for single-class held-out cohorts (all rows in this dataset),
so `N/A` is expected. Mean true-class probability is reported as the transfer-confidence fallback.

See `cross_cohort_pairwise.csv` for the full 8x8 train-on-A / test-on-B accuracy matrix and `cross_cohort_pairwise.{pdf,png}` for the heatmap.
## Deployment inference (API vs training)

Training and LOSO evaluation use **patient-level** feature rows (N participants; trials aggregated with mean, std, range, trend per trial feature). The public `POST /predict` API accepts **one trial** per request and maps it into the same column schema (`_mean` = trial value; `_std` = 0; `_range` = 0; `_trend` = NaN). That projection is **not** equivalent to multi-trial patient aggregation.

**Suggested paper wording (Limitations):**

> Deployment inference accepted one uploaded trial per API request. To match the trained feature schema, trial values populated patient-level mean columns while standard deviation and range were set to zero and trend was undefined; scores therefore do not replicate full multi-trial patient aggregation used in training (multi-trial patient aggregation). Reported confidence reflects the model maximum class probability, not external clinical calibration.

See `docs/inference_single_trial_limitation.md`. API responses include `inference_scope` and `limitations` fields.


## Validation
- Strategy: LOSO
- Classification threshold: argmax (multiclass) or Youden J per LOSO train fold (binary); see `metrics_threshold_comparison.csv` for binary baselines.
- Note: Reported metrics are intended for subject-grouped evaluation output, not in-sample prediction export.

## Model Performance (Table 2)

| Model | AUC | Accuracy | F1 | Sensitivity | p (DeLong) | p (McNemar) |
|---|---|---|---|---:|---:|---:|
| dl_bilstm_attention * | 0.929 | 0.812 | nan | nan | — | — |
| dl_gait_transformer | 0.921 | 0.765 | nan | nan | — | — |
| dl_cnn1d | 0.918 | 0.792 | nan | nan | — | — |
| dl_inception_time | 0.917 | 0.773 | nan | nan | — | — |
| dl_tcn | 0.915 | 0.815 | nan | nan | — | — |
| svm | 0.797 | 0.658 | 0.541 | 0.533 | — | — |
| ensemble_stacking | 0.796 | 0.681 | 0.582 | 0.572 | — | — |
| ensemble_soft_voting | 0.794 | 0.677 | 0.573 | 0.564 | — | — |
| xgboost | 0.790 | 0.654 | 0.526 | 0.529 | — | — |
| random_forest | 0.782 | 0.642 | 0.568 | 0.566 | — | — |
| lightgbm | 0.763 | 0.619 | 0.565 | 0.576 | — | — |
| mlp | 0.758 | 0.665 | 0.550 | 0.538 | — | — |


## Best Model
**dl_bilstm_attention**

- AUC: **0.9292**
- Accuracy: **0.8115**
- F1 Score: **nan**
- Sensitivity: **nan**

## Outputs
- table1_demographics.csv / .md / .tex
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Subject-Leakage Comparison (Grouped vs Ungrouped CV)

Compares LOSO (grouped, no subject leakage) against standard StratifiedKFold (ungrouped, permits subject leakage) to quantify the optimistic bias introduced when the same participant appears in both train and test sets.

| Model | AUC (Grouped LOSO) | AUC (Ungrouped KFold) | Inflation | Inflation % |
|---|---:|---:|---:|---:|
| xgboost | 0.7899 | 0.8040 | +0.0141 | +1.8% |
| lightgbm | 0.7634 | 0.7927 | +0.0293 | +3.8% |
| random_forest | 0.7821 | 0.7915 | +0.0094 | +1.2% |
| svm | 0.7966 | 0.8124 | +0.0158 | +2.0% |
| mlp | 0.7581 | 0.7270 | -0.0311 | -4.1% |

**Mean AUC inflation from subject leakage: +0.9%**

## Reproducibility

python main.py --config configs/pipeline_config.yaml
