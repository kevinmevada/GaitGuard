# Paper draft — title and abstract

**Journal target:** *Sensors* (MDPI)  
**Status:** Abstract draft — numeric metrics auto-synced from `metrics.csv` via `scripts/regenerate_paper_results.py` (see Metrics fill-in below).
**Keywords:** wearable sensors; inertial measurement unit; gait analysis; fall risk; machine learning; deep learning; pathology screening; leave-one-subject-out validation

---

## Working title

**Healthy-Reference Gait Anomaly Screening with Wearable IMUs: Leave-One-Subject-Out Evaluation Across Eight Clinical Cohorts (N = 260)**

*Alternates:*  
- *Unsupervised IMU Gait Anomaly Detection for Mixed Clinical Populations: 1,356 Trials, Eight Cohorts, LOSO Validation*  
- *GaitGuard: Ensemble One-Class Anomaly Screening with Supplementary Pathology-Tier Supervised Models*

---

## Abstract (five sentences — *Sensors* structure)

Falls affect approximately one in three community-dwelling adults aged ≥65 years annually and remain a leading cause of injury-related morbidity, hospitalization, and loss of independence worldwide. Wearable inertial measurement unit (IMU) gait analysis can quantify mobility impairment in clinical and ambulatory settings, yet many methods report in-sample anomaly flags or trial-level classifiers without leave-one-subject-out (LOSO) validation, omit multi-trial session aggregation, or merge heterogeneous orthopedic and neurological pathologies into a single “high-risk” label, limiting both generalizability and interpretability. We developed a reproducible pipeline on an open eight-cohort clinical dataset (N = 260 participants, 1,356 walking trials, four body-worn IMUs at 100 Hz) that extracts temporal, spectral, wavelet, trunk-dynamics, orientation, and asymmetry features and applies **primary** Healthy-reference unsupervised anomaly screening: isolation forest, local outlier factor, and one-class SVM scores are min–max normalized and averaged into a trial-level ensemble, evaluated with LOSO out-of-fold scoring against a screening pseudo-label (non-Healthy vs Healthy trials). **After full pipeline regeneration**, primary ensemble ROC-AUC and PR-AUC will be reported in **Metrics fill-in** below; supplementary supervised pathology-tier models (XGBoost, LightGBM, random forest, SVM, MLP with nested RFECV LOSO) provide secondary stratification benchmarks in `docs/paper/results.md`. Because evaluation uses cohort diagnostic membership rather than adjudicated incident falls, these results support gait **anomaly screening** as a non-invasive triage aid in mixed orthopedic–neurological populations, but **do not constitute direct fall prediction**; prospective validation with individual fall outcomes is required before clinical deployment.

---

## Metrics fill-in

_Artifacts not yet regenerated — run `python scripts/regenerate_paper_results.py` after `python main.py`._

| Metric | Value |
|--------|-------|
| Primary anomaly ensemble ROC-AUC (LOSO OOF) | _pending pipeline rerun_ |
| Primary anomaly ensemble PR-AUC | _pending_ |
| Secondary deployable ensemble macro OvR AUC | _pending pipeline rerun_ |
| Best supervised single-model LOSO macro OvR AUC | _pending pipeline rerun_ |

Regenerate after each pipeline run:

```bash
cd fall_risk_pipeline && python main.py
python ../scripts/regenerate_paper_results.py
```

Optional sensitivity sentence (binary neurological high-risk only, `label_mode: binary`, `binary_strategy: threshold_ge_2`): report a second AUC from the same table if reviewers request a dichotomous endpoint.

---

## Full paper outline (*Sensors* — quantified sections to draft next)

| Section | Content to quantify |
|---------|---------------------|
| **Introduction / background** | Fall epidemiology; IMU gait in aging and neuro/orthopedic disease; cite dataset (Figshare 10.6084/m9.figshare.28806086). |
| **Gap** | Subject leakage, trial vs patient features, label heterogeneity, lack of open reproducible LOSO benchmarks on this cohort. |
| **Methods** | N = 260, 1,356 trials, 8 cohorts, 4 IMUs, 100 Hz; preprocessing (Madgwick, gait events); temporal/spectral/wavelet/trunk/orientation/asymmetry features; RFECV cap ≤20; nested LOSO tabular eval; fixed global DL HP unless `loso_hyperparameter_tuning.enabled`; ensemble top-k soft voting; macro-OVR AUC, macro F1, per-class metrics; **Youden J** on train folds (binary); multiclass paired comparisons via bootstrap macro-OVR deltas + BH-FDR (DeLong binary-only). |
| **Results** | Table 1 demographics; primary LOSO AUC from `docs/paper/results.md` (auto-generated); calibration; SHAP top features; **feature ablation** (LOSO); anomaly vote rates by cohort; gait-event validation error (ms). |
| **Discussion / conclusion** | Screening vs diagnosis; single-trial API limitation; **limitations** (research prototype, no clinical decision support). |
| **Limitations** | **Retrospective**, **single dataset**, **no prospective follow-up**, **no ground-truth fall outcomes** (cohort labels only); internal LOSO only; path to prospective validation — `docs/limitations.md`, `docs/paper/limitations.md`. |
| **Ethics** | Public de-identified Figshare data; CPP Île-de-France II (2014-10-04 RNI); no new human data — see `docs/paper/ethics_statement.md`. |

See also: `docs/label_binning.md`, `docs/inference_single_trial_limitation.md`, `docs/ethics.md`, `fall_risk_pipeline/docs/reproducibility.md`.
