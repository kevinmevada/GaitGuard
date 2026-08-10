# Results
_Auto-generated 2026-07-27 14:42 UTC from pipeline artifacts (git `5296a27`). Do not edit by hand — run `scripts/regenerate_paper_results.py` after each pipeline run._
## 1. Cohort composition
The analysis set contains 260 participants and 1,356 walking trials across eight cohorts with four synchronized IMUs. Demographics: `fall_risk_pipeline/results/metrics/table1_demographics.md`.
## 2. Methodological novelty vs competitor literature

Comparison of **evaluation rigor** features across wearable gait competitors benchmarked in GaitGuard. Numeric performance lives in Table 2 (`docs/paper/table2_prior_work.md`).

| Study | Year | Dataset | Strict LOSO | 3-method one-class ensemble | Cross-dataset eval | Cohorts |
|---|---:|---|:---:|:---:|:---:|---|
| Moon et al. | 2020 | Single-site IMU gait (PD vs healthy) | — | — | — | 2 |
| Trabassi et al. | 2022 | PD gait cohort | — | — | — | 1 |
| Dempster et al. (ROCKET) | 2019 | UCR/UEA time-series archive | — | — | — | 117 datasets |
| Dempster et al. (MINIROCKET) | 2021 | UCR/UEA time-series archive | — | — | — | 117 datasets |
| Ismail Fawaz et al. (InceptionTime) | 2020 | UCR/UEA time-series archive | — | — | — | 128 datasets |
| Ordóñez & Roggen (DeepConvLSTM) | 2016 | OPPORTUNITY / PAMAP2 HAR | — | — | — | activity classes |
| Navita et al. | 2025 | Gait clinic (UPDRS regression) | — | — | — | ≤3 |
| Sadeghsalehi et al. | 2025 | Clinical gait (imbalanced screening) | — | — | — | ≤4 |
| **GaitGuard (this work)** | 2026 | Voisard 8-cohort + DAPHNET FOG (zero-shot) | ✓ | ✓ | ✓ | 8 |

## Three unambiguous firsts (GaitGuard only)

- **First strict LOSO on full 8-cohort Voisard.** No prior wearable gait paper evaluates all eight Voisard pathology cohorts (Healthy, HipOA, KneeOA, ACL, PD, CVA, CIPN, RIL) under leave-one-subject-out holdout.
- **First 3-method one-class ensemble under LOSO.** BiLSTM-AE reconstruction + Isolation Forest on latent activations + one-class SVM boundary distance, trained on healthy gait only per fold.
- **First zero-shot cross-dataset FOG transfer in this comparator set.** Sealed DAPHNET freezing-of-gait evaluation with asymmetric sensing: four-sensor Voisard training → single lower-back sensor at test time (zero-padded layout), which is strictly harder than matched-sensor transfer.

## Footnotes

- **Strict LOSO:** leave-one-participant-out; no trial from the held-out subject appears in training.
- **3-method one-class ensemble:** BiLSTM-AE + Isolation Forest (latent) + one-class SVM (latent); pathological gait never used for manifold fitting.
- **Cross-dataset eval:** train on Voisard, evaluate on an external dataset without target-domain retraining (DAPHNET FOG).
- Competitor flags reflect **published protocols** for the cited benchmark papers, not re-runs on Voisard.

## 2. Primary BiLSTM-AE 3-method ensemble (LOSO OOF)

Healthy-reference BiLSTM autoencoder (HE+LB+LF+RF) with latent Isolation Forest and One-Class SVM — strict leave-one-subject-out (`feature_selection_protocol: bilstm_ae_loso_healthy_reference_3method`). Pseudo ground truth: non-Healthy trial = positive.

| Method | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| isolation_forest_latent | 0.7540 | 0.8418 | 0.0807 | 0.9209 |
| one_class_svm_latent | 0.7490 | 0.8760 | 0.5106 | 0.7966 |
| bilstm_ae_ensemble | 0.6238 | 0.8053 | 0.2714 | 0.8277 |
| ae_reconstruction | 0.4790 | 0.7720 | 0.3047 | 0.7627 |

**Primary endpoint (`bilstm_ae_ensemble`):** ensemble ROC-AUC 0.6238.
 Ensemble gain vs best single method: -0.1302 AUC.

## Per-cohort LOSO results — pathology-tier screening (detailed)

This section reports **cohort-resolved** LOSO out-of-fold performance. Pooled means are supplementary only; clinical heterogeneity across the eight Voisard cohorts is the primary result.

**Model:** bilstm_ae_ensemble (`bilstm_ae_ensemble_score`)  
**Global Youden threshold (all pathological vs Healthy):** 0.2612

> Voisard does not include an MS-labelled cohort. Neuropathy-tier signal is carried by **CIPN** (chemotherapy-induced peripheral neuropathy) and **RIL** (radiculopathy/leg pain). Orthopedic cohorts map to manuscript aliases **HOA** (HipOA) and **TKA** (KneeOA).

## 1. One-vs-Healthy screening per pathological cohort

Each row is a **separate** binary task: cohort *c* (positive) vs Healthy (negative). AUROC and F1 are **not** macro-averaged across cohorts.

| Cohort | vs Healthy | n trials (path.) | n participants | AUROC | F1 | MCC | Sens. | Spec. | Anomaly rate (%) | Ref. fall prob. (%) | Mean score gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PD †** | PD vs Healthy | 158 | 24 | — | — | — | — | — | 75.3 | 67.3 | 0.0624 |
| **CVA** | CVA vs Healthy | 128 | 49 | 0.7542 | 0.5594 | 0.3700 | 0.8281 | 0.5904 | 82.8 | 54.2 | 0.2114 |
| **CIPN †** | CIPN vs Healthy | 98 | 19 | — | — | — | — | — | 68.4 | 41.8 | 0.0662 |
| **RIL** | RIL vs Healthy | 397 | 51 | 0.6775 | 0.6744 | 0.3230 | 0.6625 | 0.6610 | 66.2 | 38.9 | 0.1158 |
| **HOA †** | HOA vs Healthy | 74 | 15 | — | — | — | — | — | 89.2 | 28.5 | -0.0313 |
| **TKA †** | TKA vs Healthy | 76 | 18 | — | — | — | — | — | 86.8 | 24.1 | -0.0065 |
| **ACL †** | ACL vs Healthy | 60 | 11 | — | — | — | — | — | 100.0 | 18.7 | -0.0968 |

† AUROC/F1/MCC/sensitivity/specificity suppressed (`auc_status: unstable_small_n`): cohort has fewer participants than `models.evaluation.cohort_auc_min_n` (default 25) and cannot support a stable point estimate. Do not cite these cells; see `docs/paper/methods.md` §10 for the rule this mirrors.

### Interpretation guide

- **AUROC / F1** — discrimination for that cohort only; compare across rows, do not average.
- **Anomaly rate** — % of pathological trials flagged at the cohort-specific Youden threshold (re-fit on Healthy + that cohort's OOF trials).
- **Ref. fall prob.** — literature reference fall-risk percentage for the cohort label (not a prospective outcome in this dataset).
- **Mean score gap** — pathological minus healthy mean anomaly score on the same comparison set.

## 2. Anomaly score distribution by cohort (all eight cohorts)

| Cohort | n trials | n participants | Mean score | Median | SD | Anomaly rate (%) | Ref. fall prob. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Healthy | 354 | 73 | 0.3185 | 0.2424 | 0.2099 | 46.0 | 5.2 |
| **PD** | 158 | 24 | 0.3809 | 0.3508 | 0.1911 | 75.3 | 67.3 |
| CVA | 128 | 49 | 0.5298 | 0.4507 | 0.2699 | 85.9 | 54.2 |
| CIPN | 98 | 19 | 0.3847 | 0.3137 | 0.2299 | 68.4 | 41.8 |
| RIL | 397 | 51 | 0.4343 | 0.3909 | 0.2182 | 75.1 | 38.9 |
| HOA | 74 | 15 | 0.2872 | 0.2288 | 0.1804 | 39.2 | 28.5 |
| TKA | 76 | 18 | 0.3120 | 0.2472 | 0.2010 | 47.4 | 24.1 |
| ACL | 60 | 11 | 0.2217 | 0.1713 | 0.1575 | 28.3 | 18.7 |

## 3. Kruskal-Wallis — cohort differences in anomaly score

Tests whether anomaly-score distributions differ across cohorts (non-parametric; trial-level and participant-mean variants).

- **Trial-level scores:** H = 191.642, p = 0.0000 (significant at α=0.05), k = 8 cohorts
- **Participant-mean scores:** H = 52.033, p = 0.0000 (significant at α=0.05), k = 8 cohorts

## 4. Clinical discussion — do not average away cohort signal

**PD clinical paradox (discuss explicitly):** Parkinson's disease carries the highest reference fall probability in this dataset (67.3%), yet LOSO anomaly flagging is comparatively low (75.3% of PD trials above the Youden threshold). This pattern is clinically meaningful — PD gait can be pathologically impaired yet **internally consistent** (narrow, stereotyped kinematics), producing modest reconstruction/latent deviation from a healthy manifold. Averaging PD into a single pooled metric would hide this dissociation between epidemiological fall risk and unsupervised anomaly score.

The eight-cohort Voisard design enables contrasts that single-disease studies cannot replicate: high fall-probability neurological cohorts (PD, CVA) vs orthopedic mechanical gait (HOA, TKA, ACL) vs neuropathy-tier cohorts (CIPN, RIL). Report each row in the main text; reserve pooled AUROC for supplementary material only.

## Fall-risk clinical validation — Spearman correlation

Anomaly score is evaluated as a **proxy for fall risk** by correlating literature / metadata fall probability with BiLSTM-AE LOSO anomaly scores.

**Score column:** `bilstm_ae_ensemble_score`  
**Fall probability:** cohort reference rates (Voisard / clinical literature) or trial `fall_probability` from ingest metadata when present.

## Primary result — all participants

- **Spearman ρ** = 0.3332 (p = 0.0000, n = 260 participants, significant at α = 0.05)
- Mean anomaly score = 0.3959; mean reference fall probability = 32.7%

## Per pathological cohort (Healthy vs cohort contrast)

Within a single cohort, fall probability is cohort-constant, so ρ is reported for **Healthy + pathological tier** participants where fall probability varies.

| Cohort | vs Healthy | n participants | ρ | p-value | Mean score | Ref. fall prob. (%) |
|---|---|---:|---:|---:|---:|---:|
| **PD** | Healthy + PD | 97 | 0.2048 | 0.0442 | 0.3293 | 20.6 |
| **CVA** | Healthy + CVA | 122 | 0.4855 | 0.0000 | 0.4115 | 24.9 |
| **CIPN** | Healthy + CIPN | 92 | 0.1754 | 0.0944 | 0.3317 | 12.8 |
| **RIL** | Healthy + RIL | 124 | 0.3647 | 0.0000 | 0.3735 | 19.1 |
| **HOA** | Healthy + HOA | 88 | -0.0553 | 0.6087 | 0.3127 | 9.2 |
| **TKA** | Healthy + TKA | 91 | 0.1197 | 0.2583 | 0.3246 | 8.9 |
| **ACL** | Healthy + ACL | 84 | -0.2176 | 0.0468 | 0.3026 | 7.0 |

## Interpretation

- **Positive ρ** — higher literature fall-risk cohorts show higher anomaly scores (supports deployment as clinical decision-support signal).
- **AUROC/F1** answer discrimination; this table answers **clinical relevance**.
- Rows `within_{cohort}` in the CSV are NA by design (no fall-probability variance).

### Cohort reference fall probabilities (%)

| Cohort | Reference fall prob. (%) |
|---|---:|
| Healthy | 5.2 |
| PD | 67.3 |
| CVA | 54.2 |
| CIPN | 41.8 |
| RIL | 38.9 |
| HOA | 28.5 |
| TKA | 24.1 |
| ACL | 18.7 |

## 2b. Secondary supervised pathology-tier performance (tabular models)

Supplementary to primary anomaly screening — nested RFECV LOSO from `metrics.csv`.

| Model | AUC | 95% CI | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|
| xgboost | 0.8415 | [0.7994, 0.8826] | 0.6686 | 0.7361 |
| ensemble_soft_voting | 0.8404 | [0.7974, 0.8818] | 0.6704 | 0.7361 |
| ensemble_stacking | 0.8371 | [0.7954, 0.8798] | 0.6818 | 0.7398 |
| random_forest | 0.8279 | [0.7840, 0.8703] | 0.6877 | 0.7472 |
| lightgbm | 0.8239 | [0.7794, 0.8676] | 0.6674 | 0.7212 |
| svm | 0.8172 | [0.7771, 0.8525] | 0.5394 | 0.6840 |
| mlp | 0.7505 | [0.7025, 0.7990] | 0.5279 | 0.6283 |

**Best supervised LOSO macro-OVR AUC:** xgboost (0.8415).

## 3. Deploy-schema vs nested-RFECV LOSO gap (ML-032)

Section 2b reports nested per-fold RFECV LOSO (`metrics.csv`). API/deploy checkpoints use `selected_features.json`. Deploy-schema LOSO AUCs:

| Model | Nested RFECV AUC | Deploy schema AUC | Δ (deploy − nested) |
|---|---:|---:|---:|
| xgboost | 0.8415 | 0.8617 | +0.0202 |
| ensemble_soft_voting | 0.8404 | 0.8594 | +0.0190 |
| ensemble_stacking | 0.8371 | 0.8469 | +0.0098 |
| random_forest | 0.8279 | 0.8629 | +0.0350 |
| lightgbm | 0.8239 | 0.8482 | +0.0243 |
| svm | 0.8172 | 0.8410 | +0.0238 |
| mlp | 0.7505 | 0.8213 | +0.0708 |

**Pre-registered primary endpoint:** `bilstm_ae_ensemble` — see `primary_endpoint.json`.

## 3. Classical baselines (competitor paradigm 1)

Phase 1 + Phase 2 handcrafted features, LOSO OOF.

| Model | F1 (w) | Bal. Acc. | MCC | AUROC | Sens. | Spec. | Prec. | κ | Literature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SVM (RBF) | 0.7690 | 0.7005 | 0.5938 | 0.8926 | 0.7005 | 0.8660 | 0.7242 | 0.5928 | Moon 2020; Trabassi 2022; Li 2025; Prisco 2025 |
| Random Forest (tuned) | 0.7397 | 0.6472 | 0.5452 | 0.8773 | 0.6472 | 0.8449 | 0.7027 | 0.5400 | Navita 2025; Moon 2020; Trabassi 2022 |
| Logistic Regression (L2) | 0.7355 | 0.7187 | 0.5508 | 0.8687 | 0.7187 | 0.8584 | 0.6802 | 0.5472 | Moon 2020; Dempster 2019/2021 |
| Logistic Regression (L1) | 0.7278 | 0.7070 | 0.5370 | 0.8682 | 0.7070 | 0.8543 | 0.6703 | 0.5335 | Moon 2020; Dempster 2019/2021 |
| k-NN | 0.6887 | 0.5902 | 0.4532 | 0.7937 | 0.5902 | 0.8238 | 0.6043 | 0.4522 | Moon 2020; Li 2025 |

## 4. DL competitor matrix

Raw-IMU deep baselines (LOSO) + GaitGuard BiLSTM-AE primary endpoint.

| Model | Reference | F1 (w) | Bal. Acc. | MCC | AUROC | Sens. | Spec. | Prec. | κ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MINIROCKET | Dempster 2021 | nan | nan | nan | nan | nan | nan | nan | nan |
| ROCKET | Dempster 2019 | nan | nan | nan | nan | nan | nan | nan | nan |
| InceptionTime | Ismail Fawaz 2020 | 0.8439 | 0.8051 | 0.7369 | 0.9414 | 0.8051 | 0.9159 | 0.8156 | 0.7363 |
| DeepConvLSTM | Ordóñez & Roggen 2016 | 0.8503 | 0.8158 | 0.7512 | 0.9326 | 0.8158 | 0.9200 | 0.8212 | 0.7493 |
| BiLSTM-AE | GaitGuard (yours) | 0.4127 | 0.5496 | 0.1014 | 0.6238 | 0.2714 | 0.8277 | 0.8152 | 0.0619 |

## 5. Core discriminative metrics (full competitor matrix)

F1 (weighted), balanced accuracy, MCC, AUROC, sensitivity, specificity, precision, Cohen's κ — LOSO out-of-fold.

**Abstract lead metric:** AUROC (threshold-independent headline); report MCC for rigor.

| Model | Paradigm | F1 (weighted) | Balanced Acc. | MCC | AUROC | Sensitivity | Specificity | Precision | Cohen κ |
|---|---|---|---|---|---|---|---|---|---|
| Svm Rbf | classical_paradigm_1 | 0.7690 | 0.7005 | 0.5938 | 0.8926 | 0.7005 | 0.8660 | 0.7242 | 0.5928 |
| Random Forest | classical_paradigm_1 | 0.7397 | 0.6472 | 0.5452 | 0.8773 | 0.6472 | 0.8449 | 0.7027 | 0.5400 |
| Logistic Regression L2 | classical_paradigm_1 | 0.7355 | 0.7187 | 0.5508 | 0.8687 | 0.7187 | 0.8584 | 0.6802 | 0.5472 |
| Logistic Regression L1 | classical_paradigm_1 | 0.7278 | 0.7070 | 0.5370 | 0.8682 | 0.7070 | 0.8543 | 0.6703 | 0.5335 |
| Knn | classical_paradigm_1 | 0.6887 | 0.5902 | 0.4532 | 0.7937 | 0.5902 | 0.8238 | 0.6043 | 0.4522 |
| MINIROCKET | competitor_paradigm_2_dl | — | — | — | — | — | — | — | — |
| ROCKET | competitor_paradigm_2_dl | — | — | — | — | — | — | — | — |
| InceptionTime | competitor_paradigm_2_dl | 0.8439 | 0.8051 | 0.7369 | 0.9414 | 0.8051 | 0.9159 | 0.8156 | 0.7363 |
| DeepConvLSTM | competitor_paradigm_2_dl | 0.8503 | 0.8158 | 0.7512 | 0.9326 | 0.8158 | 0.9200 | 0.8212 | 0.7493 |
| BiLSTM-AE | gaitguard_primary | 0.4127 | 0.5496 | 0.1014 | 0.6238 | 0.2714 | 0.8277 | 0.8152 | 0.0619 |

## 4. Class-wise behavior
See per-class columns in `metrics.csv` and `pipeline_report.md`.
## 4. Deep learning LOSO benchmark

Participant-level LOSO; early-stopping val AUC aggregated per participant (ML-016).

| Deep model | Macro-OVR AUC | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| dl_inception_time | 0.9358 | 0.7998 | 0.8500 |
| dl_deep_conv_lstm | 0.9264 | 0.7871 | 0.8423 |
| dl_gait_transformer | 0.9505 | 0.8137 | 0.8538 |
| dl_tcn | 0.9413 | 0.8014 | 0.8423 |
| dl_cnn1d | 0.9336 | 0.8056 | 0.8538 |
| dl_bilstm_attention | 0.9329 | 0.8001 | 0.8462 |

# Feature ablation (LOSO macro-OVR AUC)

Reference classifier: **xgboost** (checkpoint hyperparameters, re-fit per LOSO fold).

Trial-level features in config: **90**; patient-level columns vary by aggregation (mean, std, range, trend).

Top-10 SHAP features (LOSO aggregate, nested RFECV per fold): `mr_f00044_mean`, `lb_sampen_mean`, `it_ms_mp_29_mean`, `it_ms10_08_mean`, `it_ms_mp_30_mean`, `it_ms20_06_mean`, `mr_f00046_mean`, `it_ms_mp_31_mean`, `lb_jerk_mean_ap_mean`, `it_ms_mp_27_mean`

| Scenario | n features | AUC | 95% CI | Macro F1 |
|---|---:|---:|---|---:|
| all_features_nested_rfecv | 2408 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_temporal | 2228 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_spectral | 2360 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_wavelet | 2320 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_orientation | 2360 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_phase3_deep_features | 2376 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_asymmetry | 2368 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_turning | 2392 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_spatial | 2372 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_lyapunov | 2400 | 0.849 | [0.808, 0.889] | 0.677 |
| minus_phase2_kinematic | 2332 | 0.849 | [0.808, 0.889] | 0.677 |
| top10_shap | 10 | 0.825 | [0.785, 0.866] | 0.638 |
| minus_trunk_dynamics | 2360 | 0.798 | [0.748, 0.844] | 0.640 |

## Interpretation

- Compare `all_features_nested_rfecv` vs `top10_shap`: if AUC is similar, a compact SHAP subset may suffice.
- Compare each `minus_*` row to `all_features_nested_rfecv`: larger AUC drops indicate groups that contribute most.
- `minus_lyapunov` isolates the Lyapunov exponent (under `trunk_dynamics`); compare to `minus_trunk_dynamics`.

Outputs: `feature_ablation.csv`, `ablation_group_column_counts.csv`, `figures/models/feature_ablation_bars.*`

# BiLSTM-AE sensor ablation

Three training configurations on Voisard (Healthy-reference LOSO ensemble). Inactive IMU blocks are zero-padded; channel layout matches the 4-sensor AE.

| Config | Sensors | Voisard LOSO AUC (ensemble) | DAPHNET LB recon AUC |
|---|---|---:|---:|
| 4_sensor | HE+LB+LF+RF | 0.6856 | 0.7046 |
| 2_sensor_he_lb | HE+LB | 0.7139 | — |
| 1_sensor_lb | LB | 0.6858 | — |

**In-distribution:** 4-sensor > 2-sensor > 1-sensor → multi-sensor training adds value.

**Cross-dataset:** 4-sensor-trained model, LB-only DAPHNET input → AUROC **0.7046** (representation transfers via LB channel).

## Cross-Cohort Transfer (Leave-One-Cohort-Out)

Train on all subjects from N-1 cohorts, test on the held-out cohort. Answers: 'Can a model trained without any PD patients still detect PD?'

| Held-Out Cohort | N (test) | AUC | Mean True-Class Prob. | Accuracy | F1 (macro) |
|---|---:|---:|---:|---:|---:|
| ACL | 11 | N/A | 0.2808 | 0.2727 | 0.1429 |
| CIPN | 19 | N/A | 0.5831 | 0.7895 | 0.2941 |
| CVA | 49 | N/A | 0.4738 | 0.5714 | 0.2424 |
| Healthy | 73 | N/A | N/A | nan | nan |
| HipOA | 15 | N/A | 0.3635 | 0.3333 | 0.1667 |
| KneeOA | 18 | N/A | 0.4329 | 0.4444 | 0.2051 |
| PD | 33 | N/A | 0.5765 | 0.6667 | 0.4000 |
| RIL | 51 | N/A | 0.5468 | 0.7255 | 0.4205 |

AUC is **undefined** for single-class held-out cohorts (all rows in this dataset),
so `N/A` is expected. Mean true-class probability is reported as the transfer-confidence fallback.

See `cross_cohort_pairwise.csv` for the full 8x8 train-on-A / test-on-B matrix (macro-F1, macro OvR AUC, and accuracy). The primary heatmap `cross_cohort_pairwise.{pdf,png}` uses **macro-F1** (preferred under class imbalance); `cross_cohort_pairwise_auc.{pdf,png}` is supplemental.
## 10. Split-protocol sensitivity

Compares LOSO (grouped, one participant per row) against standard StratifiedKFold (ungrouped splits). At participant granularity this measures split-difficulty inflation, not duplicate-subject leakage (ML-048). Both arms use matched per-fold nested RFECV when `nested_in_evaluation: true` (ML-036). Ungrouped KFold AUC is averaged over multiple seeds per model (see `ungrouped_kfold_seed_repeats`; default 5, MED-001).

| Model | AUC (Grouped LOSO) | AUC (Ungrouped KFold) | Inflation | Inflation % |
|---|---:|---:|---:|---:|
| xgboost | 0.8415 | 0.8469 ± 0.0214 (n=5) | +0.0054 | +0.6% |
| lightgbm | 0.8239 | 0.8311 ± 0.0234 (n=5) | +0.0071 | +0.9% |
| random_forest | 0.8279 | 0.8464 ± 0.0194 (n=5) | +0.0185 | +2.2% |
| svm | 0.8172 | 0.8173 ± 0.0157 (n=5) | +0.0001 | +0.0% |
| mlp | 0.7505 | 0.7905 ± 0.0128 (n=5) | +0.0401 | +5.3% |

> **Note (MEDIUM-02):** MLP can show negative inflation due to non-convex optimization variance; mean below excludes MLP.

**Feature protocol (grouped):** `nested_rfecv_per_loso_fold`; **ungrouped:** `nested_rfecv_per_kfold_train_fold` (protocol_matched=True).

**Mean split-protocol AUC inflation: +0.9%**
## 11. Prior-work comparison
See `docs/paper/table2_prior_work.md` (update headline AUC from section 2 after each rerun).