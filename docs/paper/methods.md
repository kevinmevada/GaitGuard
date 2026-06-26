# Methods

## 1. Study design and objective

This work is a retrospective secondary analysis of a public, de-identified wearable IMU gait dataset. The **primary** objective is Healthy-reference **gait anomaly screening**: trial-level unsupervised ensemble scores (isolation forest, local outlier factor, one-class SVM) evaluated with leave-one-subject-out (LOSO) out-of-fold scoring. Screening pseudo ground truth for evaluation metrics is non-Healthy vs Healthy trial labels (cohort proxy, not verified falls). **Secondary** supervised pathology-tier models (healthy, orthopedic, neurological) provide supplementary stratification benchmarks and must not be pooled with anomaly metrics in a single primary performance table. The analysis is implemented as an end-to-end reproducible pipeline with fixed configuration, deterministic seeding, and stage-wise artifact generation.

## 2. Dataset and cohorts

We used the Springer Nature Figshare dataset of clinical gait signals (DOI: 10.6084/m9.figshare.28806086), comprising 1,356 walking trials from 260 participants sampled at 100 Hz with four IMUs (head, lower back, left foot, right foot). Cohorts included Healthy, Hip OA, Knee OA, ACL, PD, CVA, CIPN, and RIL. Cohort labels were mapped to pathology-tier targets:

- Class 0: Healthy
- Class 1: Orthopedic (Hip OA, Knee OA, ACL)
- Class 2: Neurological (PD, CVA, CIPN, RIL)

These are cohort-level diagnostic categories; participant-level prospective fall outcomes were not available.

### 2.1 DAPHNET cross-dataset extension (PD freezing gait)

We additionally ingested the public DAPHNET freezing-gait corpus (64 Hz, ankle / thigh / trunk accelerometry; 17 recordings, 10 PD participants) for **zero-shot** transfer evaluation against models trained on the four-sensor Voisard corpus.

**Sensor mapping is a design decision, not a preprocessing step.** DAPHNET trunk → Voisard lower back (LB) is the only anatomically defensible pairing: lumbar placement, shared dominant axes during gait, and the same trunk-dynamics phenomenon captured by LB features in the primary pipeline. DAPHNET ankle was **not** mapped to Voisard left/right foot (lateral malleolus vs dorsal foot; different heel-strike vibration). DAPHNET thigh has no Voisard equivalent and was **dropped**. After calibration-row removal, trunk signals were resampled to 100 Hz with `resample_poly(up=25, down=16)` and stored under the Voisard `lower_back` channel key.

**Single-sensor zero-shot evaluation.** Primary Voisard models train on four IMUs (head, LB, left foot, right foot), learning a rich multi-site Healthy reference. At DAPHNET evaluation time, only LB input is provided—the sole shared channel. A zero-shot AUROC above 0.77 on this harder single-sensor transfer supports both (1) multi-sensor training improving the reference representation and (2) cross-dataset generalization without forced anatomical alignment. This is a stricter claim than same-configuration cross-dataset transfer; most prior wearable-gait studies do not attempt cross-dataset evaluation at all.

**FOG label mapping (eval-only).** Per-sample DAPHNET annotations map to sealed-test ground truth: `annotation == 1` → `y_true = 0` (normal walking); `annotation == 2` → `y_true = 1` (freezing of gait). Labels are written to `data/processed/daphnet/fog_labels.npz` as separate NumPy arrays per subject — **never** concatenated into feature tensors or used during Voisard model training. The headline DAPHNET metric is `roc_auc_score(y_true, anomaly_scores)` on LB-only zero-shot anomaly scores (Healthy-reference ensemble fit on Voisard trunk accelerometry only). This sealed test runs once via the `anomaly` stage (`results/metrics/daphnet_fog_auroc.json`); do not tune anomaly models against it.

Manifest: `results/metrics/daphnet_sensor_mapping.json`; per-trial fields `source_dataset`, `sensor_mapping`, `eval_sensors` in `trial_metadata.csv`.

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

- **Stage C unified bandpass (Voisard + DAPHNET @ 100 Hz):** 4th-order Butterworth bandpass 0.5–20 Hz on all accelerometer channels via zero-phase `filtfilt` (no group-delay peak shift on stride events). Applied identically to both corpora after DAPHNET resampling; any asymmetry would invalidate cross-dataset comparison.
- Optional Madgwick orientation fusion (head and lower-back sensors; Voisard)
- Gravity-removal path for trunk acceleration feature computation
- Algorithmic gait-event detection from foot signals
- Trial-level validity checks including minimum trial duration

A dedicated gait-event validation stage compared detected heel strikes with dataset annotations within a tolerance window and exported precision/recall-style quality summaries.

## 5. Feature engineering

Trial-level features were extracted from multi-sensor signals across these domains:

- Temporal/gait-cycle metrics (e.g., stride timing, cadence, stance/swing characteristics)
- Spectral features (including dominant frequency, spectral centroid/entropy, band powers)
- Wavelet sub-band energy, energy ratios, and wavelet entropy (DWT decomposition)
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

To preserve mechanistically relevant nonlinear dynamics in final training, up to **four** nonlinear dynamics features (`sampen` / `dfa` family) were retained regardless of RFECV rank to ensure mechanistic coverage (MED-005 / ML-040). When the family exceeds `max_required_features`, candidates are ranked by mean |SHAP| before the cap is applied; mean |SHAP| values for forced vs dropped candidates are reported in `required_feature_shap_audit.csv`. **Sensitivity check:** re-run with `required_feature_substrings: []` and compare LOSO AUC — if forced features reduce performance, interpret mechanistic claims cautiously; if neutral or beneficial, the constraint is data-supported.

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

Deep evaluation used leave-one-subject-out (LOSO) design with per-subject holdout. Each LOSO train fold held out ~10% of remaining participants for inner validation (stratified participant-level split; no window leakage). Inner-val participant draws use a fold-scoped RNG seed (`base_seed + fold_idx × 31337`) so early-stopping validation does not repeatedly anchor on the same subjects across nearby folds (HIGH-002). Mixed precision (AMP) was enabled on CUDA when available. Probability outputs were numerically stabilized before metric computation (finite-value guard, clipping, and row normalization) to ensure valid multiclass AUC evaluation.

Hyperparameter protocol (ML-042 / HIGH-001): tabular models re-tune with Optuna on each LOSO train fold (10 trials per fold by default). Deep models run an analogous **per-LOSO-fold learning-rate search**: Optuna (`n_trials: 5`, `search_epochs: 12`) on the inner participant validation split held out from the outer LOSO train fold, then full training with the selected rate and global `max_epochs` / early stopping. Set `deep_learning.loso_hyperparameter_tuning.enabled: false` only for fast/debug runs (introduces asymmetry vs tabular). Exported `deep_learning_metrics.csv` includes `hyperparameter_protocol` (`fixed_global_config` vs `loso_inner_participant_optuna_lr`).

Window overlap protocol (ML-043): trials are sliced with 50% overlap (`overlap: 0.5`) to increase held-out participant coverage for participant-level soft voting at inference. Overlapping windows are highly correlated; by default, inner train/validation use `training_window_deduplication: true`, retaining at most one window per non-overlapping stride block per trial. Participant-balanced CE weights and participant-level validation AUC / soft-vote aggregation further reduce window-level pseudo-replication. Metrics export `window_overlap`, `training_window_protocol`, and `inference_window_protocol`.

## 9. Evaluation protocol and metrics

### 9.1 Primary protocol

**Subject-grouped splitting (no trial-level leakage).** All trials are keyed by `participant_id` before any train/validation/test assignment. We never randomize individual trials across splits: every trial from a given participant stays in the same partition. For Healthy-reference holdout bookkeeping, Healthy participants are split **70% train / 15% validation / 15% test** (seeded partition of the subject list); **all pathological participants are assigned to test only**. Exported manifests: `subject_split.csv` / `subject_split.json`. Before model fitting, the pipeline asserts `set(train_ids) ∩ set(test_ids) == ∅` and aborts with `DATA LEAKAGE: subject in both splits` if violated.

**LOSO evaluation** holds out one participant per fold; all trials from that subject form the test fold while remaining subjects supply training data. This is a leave-one-**subject**-out split by participant ID — no subject appears in both train and test within a fold, unlike Klaver et al. (2023), who randomized trials and were criticized in review when the same subject's gait signature appeared in both train and test (inflated AUC). Inner hyperparameter validation (tabular Optuna, deep learning learning-rate search) uses additional **participant-level** holdouts (`StratifiedGroupKFold` / stratified participant splits), never window- or trial-level randomization.

**Primary endpoint (`anomaly_ensemble`):** trial-level Healthy-reference anomaly screening with LOSO out-of-fold evaluation (ANOM-001). For each held-out participant, one-class models (isolation forest, LOF, one-class SVM) were fit on Healthy training trials only; held-out trials received normalized decision scores averaged into an ensemble. Evaluation pseudo-label: non-Healthy trial = positive (screening target). Youden J thresholds were fit on OOF scores for sensitivity/specificity reporting. Artifacts: `anomaly_metrics.csv`, `anomaly_threshold.json`, `primary_endpoint.json`.

**Secondary supervised protocol:** participant-grouped nested RFECV LOSO for pathology-tier tabular models (deploy-schema ensemble for API parity). This must not be pooled with primary anomaly metrics in a single headline table.

### 9.2 Primary and secondary metrics

Primary metrics (anomaly ensemble, LOSO OOF):

- ROC-AUC (non-Healthy vs Healthy pseudo-label)
- PR-AUC
- Sensitivity / specificity at Youden J threshold

Secondary supervised metrics (pathology-tier tabular models):

- Macro one-vs-rest AUC (multiclass)
- Macro F1
- Accuracy
- Sensitivity/specificity summaries
- Confusion matrices
- Calibration diagnostics (where enabled)

For discrete operating-point metrics, thresholds were derived from training data (Youden J strategy) and applied to held-out test folds.

### 9.3 Statistical comparison

Paired model-comparison procedures included:

- **Binary label mode:** DeLong-style paired AUC testing and McNemar testing on paired categorical predictions (exact binomial McNemar via `mcnemar_exact: true`; chi-squared approximation is invalid when discordant pairs ≪ 25 — HIGH-004).
- **Multiclass label mode (primary):** Bootstrap paired macro-OVR AUC differences with Benjamini–Hochberg FDR correction; DeLong is not applied to multiclass OvR scores. McNemar tests use argmax OOF class predictions (correct-vs-wrong discordant pairs; exact binomial test, exploratory).

All tests use aligned out-of-fold predictions to maintain pairing validity.

## 10. Explainability and robustness analyses

Participant composition (per-cohort *n*, age mean ± SD, sex ratio) is reported in **Table 1** (`table1_demographics.csv` / `.md` / `.tex`), generated from `trial_metadata.csv` at ingest/report time (MED-003).

For held-out cohorts with single-class test composition, AUC was marked undefined and accompanied by fallback confidence-oriented reporting fields. Cohort subgroup rows in `metrics_by_cohort.csv` with `n < cohort_auc_min_n` (default **25**) export `auc_status: unstable_small_n` and leave AUC / AUC CI / AUC-PR blank (NaN) so small-n point estimates are not cited as stable (ML-044 / MED-003). *Footnote for cohort AUC tables: cohorts with n < 25 participants — AUC suppressed due to insufficient sample size for stable estimation.*

Explainability:

- SHAP-based global feature importance from LOSO/out-of-fold aggregation
- Per-model and cohort-aware importance exports

Exploratory data analysis (EDA) figures include a participant-level t-SNE embedding. Features were standardized with `StandardScaler` fit on **all** participants before t-SNE (visualization only; not used in model training). The figure caption and `tsne_caption.txt` note that all supervised evaluation uses per-fold normalization (MED-004).

Robustness:

- Feature ablation (all features vs reduced groups)
- Sensor ablation (single-sensor through multi-sensor subsets)
- Cross-cohort transfer evaluation (train on N-1 cohorts, test on held-out cohort)

Protocol disclosure: feature and sensor ablations use **leave-one-subject-out (LOSO)** with **per-fold nested RFECV** intersected with each scenario column mask (`nested_in_ablation: true`). Primary Table 2 tabular LOSO uses the same nested RFECV protocol on the full feature matrix. Ablation AUCs rank feature/sensor subsets within that shared protocol but remain exploratory (different column sets per scenario).

## 11. Anomaly screening (primary endpoint)

Trial-level Healthy-reference anomaly screening used three one-class detectors:

- Isolation Forest (`contamination=0.1`, 100 trees)
- Local Outlier Factor (`novelty=True`, adaptive `n_neighbors`)
- One-Class SVM (RBF kernel, `nu=0.1`)

`StandardScaler` was fit on Healthy training rows only. Per-method decision scores were min–max normalized (Healthy-reference range per LOSO fold) and averaged into an ensemble score. Deploy/API inference uses full-cohort Healthy-fit scalers with deploy-time calibration ranges (`deploy_calibration.json`) and LOSO Youden cutoff (`anomaly_threshold.json`). A majority-vote rule (≥2 of 3 methods) is retained as a secondary binary flag for interpretability.

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
