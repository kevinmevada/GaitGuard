# Methods

## 1. Study design and objective

This work is a retrospective secondary analysis of a public, de-identified wearable IMU gait dataset. The objective is pathology-tier gait screening using three supervised classes (healthy, orthopedic, neurological) as proxies for fall-risk stratification in mixed clinical populations. The analysis is implemented as an end-to-end reproducible pipeline with fixed configuration, deterministic seeding, and stage-wise artifact generation.

## 2. Dataset and cohorts

We used the Springer Nature Figshare dataset of clinical gait signals (DOI: 10.6084/m9.figshare.28806086), comprising 1,356 walking trials from 260 participants sampled at 100 Hz with four IMUs (head, lower back, left foot, right foot). Cohorts included Healthy, Hip OA, Knee OA, ACL, PD, CVA, CIPN, and RIL. Cohort labels were mapped to pathology-tier targets:

- Class 0: Healthy
- Class 1: Orthopedic (Hip OA, Knee OA, ACL)
- Class 2: Neurological (PD, CVA, CIPN, RIL)

These are cohort-level diagnostic categories; participant-level prospective fall outcomes were not available.

## 3. Pipeline overview

The pipeline executes sequential stages:

1. Ingestion
2. Gait-event validation
3. Signal preprocessing
4. Exploratory data analysis
5. Feature extraction
6. Feature selection
7. Tabular model training
8. Tabular model evaluation
9. Deep model LOSO training/evaluation
10. Feature ablation
11. Sensor ablation
12. Cross-cohort transfer
13. Prediction export
14. Anomaly analysis
15. Report generation

The primary configuration file is `fall_risk_pipeline/configs/pipeline_config.yaml`.

## 4. Signal preprocessing

Preprocessing used configuration-controlled steps:

- Butterworth filtering (band-limited denoising for gait-relevant frequencies)
- Optional Madgwick orientation fusion (head and lower-back sensors)
- Gravity-removal path for trunk acceleration feature computation
- Algorithmic gait-event detection from foot signals
- Trial-level validity checks including minimum trial duration

A dedicated gait-event validation stage compared detected heel strikes with dataset annotations within a tolerance window and exported precision/recall-style quality summaries.

## 5. Feature engineering

Trial-level features were extracted from multi-sensor signals across these domains:

- Temporal/gait-cycle metrics (e.g., stride timing, cadence, stance/swing characteristics)
- Spectral features (including dominant frequency, spectral centroid/entropy, band powers)
- Trunk dynamics (RMS, jerk, Lyapunov, approximate entropy, sample entropy, DFA)
- Orientation/postural features (tilt/pitch/roll variability and sway surrogates)
- Asymmetry features (between-limb timing and dynamic imbalance indicators)
- Turning descriptors

Patient-level matrices were then formed by within-session aggregation of each trial feature using mean, standard deviation, range, and linear trend across ordered trials.

### 5.1 Current artifact-state disclosure

The committed pipeline configuration includes nonlinear feature families (`lb_sampen`, `lb_dfa`, `head_sampen`, `head_dfa`, `head_lb_dfa_ratio`) and feature-selection retention rules for `sampen`/`dfa`. However, the currently committed performance artifacts (`results/metrics/metrics.csv`, SHAP exports, and ablation summaries) were produced before a full end-to-end rerun that propagates those families through all downstream stages. Therefore, manuscript interpretation should treat nonlinear-feature claims as **configured methodology** pending **full rerun confirmation** in final reported results.

## 6. Feature selection

To control dimensionality and improve generalization, feature selection was applied before final model fitting:

- Primary selector: grouped RFECV (participant-aware folds)
- Secondary ranking: SHAP-based importance pruning
- Export cap: <=20 selected features (configurable)

To preserve mechanistically relevant nonlinear dynamics in final training, required feature-family retention rules were enforced for specified substrings (including `sampen` and `dfa`) when configured. Final manuscript numbers should only claim retained nonlinear families after rerun-generated model metrics and SHAP outputs confirm their propagation into trained checkpoints and evaluation exports.

## 7. Tabular models and ensemble

The tabular branch trained and tuned the following models:

- XGBoost
- LightGBM
- Random Forest
- SVM
- MLP

Hyperparameter optimization used Optuna under grouped cross-validation constraints. A soft-voting ensemble combined top-performing base learners for the primary tabular endpoint.

## 8. Deep learning models

The deep branch benchmarked five architectures:

- InceptionTime
- Gait Transformer
- TCN
- CNN-1D
- BiLSTM with attention

Deep evaluation used leave-one-subject-out (LOSO) design with per-subject holdout. Mixed precision (AMP) was enabled on CUDA when available. Probability outputs were numerically stabilized before metric computation (finite-value guard, clipping, and row normalization) to ensure valid multiclass AUC evaluation.

## 9. Evaluation protocol and metrics

### 9.1 Primary protocol

Primary performance estimation used participant-grouped validation (LOSO for deep models; grouped/nested scheme for tabular evaluation paths). This minimizes participant overlap risk between training and testing folds and was paired with an explicit grouped-vs-ungrouped sensitivity audit.

### 9.2 Primary and secondary metrics

Primary metric:

- Macro one-vs-rest AUC (multiclass)

Secondary metrics:

- Macro F1
- Accuracy
- Sensitivity/specificity summaries
- Confusion matrices
- Calibration diagnostics (where enabled)

For discrete operating-point metrics, thresholds were derived from training data (Youden J strategy) and applied to held-out test folds.

### 9.3 Statistical comparison

Paired model-comparison procedures included:

- DeLong-style paired AUC testing
- Bootstrap-based paired comparisons
- McNemar testing on paired categorical predictions

These tests were performed on aligned out-of-fold predictions to maintain pairing validity.

## 10. Explainability and robustness analyses

Explainability:

- SHAP-based global feature importance from LOSO/out-of-fold aggregation
- Per-model and cohort-aware importance exports

Robustness:

- Feature ablation (all features vs reduced groups)
- Sensor ablation (single-sensor through multi-sensor subsets)
- Cross-cohort transfer evaluation (train on N-1 cohorts, test on held-out cohort)

Protocol disclosure: unlike the primary tabular/deep evaluation paths, sensor ablation is run with **StratifiedGroupKFold (5 folds)** participant-grouped CV (not full LOSO) for computational tractability across all sensor-subset combinations. Sensor-ablation AUCs are grouped-CV estimates (5-fold StratifiedGroupKFold) rather than full LOSO, making them appropriate for within-experiment sensor-subset ranking but not directly comparable to primary LOSO AUC estimates. The ranking finding (head+right foot >= full 4-sensor) is robust to this protocol difference.

For held-out cohorts with single-class test composition, AUC was marked undefined and accompanied by fallback confidence-oriented reporting fields.

## 11. Anomaly analysis

In addition to supervised screening, unsupervised anomaly analysis was performed using:

- Isolation Forest
- Local Outlier Factor
- One-Class SVM

A majority-vote decision rule (>=2 positive anomaly votes) generated a companion anomaly flag. This module is supplemental and does not replace supervised pathology-tier evaluation.

## 12. Reproducibility and implementation

Implementation language and core stack:

- Python
- scikit-learn, XGBoost, LightGBM, PyTorch
- Optuna, SHAP, NumPy, pandas, SciPy

Reproducibility controls:

- Fixed pipeline seeds via configuration
- Deterministic torch option
- Versioned YAML configuration
- Stage-wise artifact outputs (metrics, figures, checkpoints, reports)
- Root-level Docker/Make automation for one-command reruns

All analyses were performed on local compute with automatic CUDA selection when available; CPU fallback was supported.

## 13. Ethical and reporting boundary

This analysis used only previously published de-identified data (no new recruitment). Model outputs are intended for research screening context and should not be interpreted as prospective, clinically validated fall prediction without external prospective outcome-based validation.
