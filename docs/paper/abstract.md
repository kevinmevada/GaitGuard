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

Falls affect approximately one in three community-dwelling adults aged ≥65 years annually and remain a leading cause of injury-related morbidity, hospitalization, and loss of independence worldwide. Wearable inertial measurement unit (IMU) gait analysis can quantify mobility impairment in clinical and ambulatory settings, yet many methods report in-sample anomaly flags or trial-level classifiers without leave-one-subject-out (LOSO) validation, omit multi-trial session aggregation, or merge heterogeneous orthopedic and neurological pathologies into a single “high-risk” label, limiting both generalizability and interpretability. We developed a reproducible pipeline on an open eight-cohort clinical dataset (N = 260 participants, 1,356 walking trials, four body-worn IMUs at 100 Hz) that extracts temporal, spectral, wavelet, trunk-dynamics, orientation, and asymmetry features and applies **primary** Healthy-reference unsupervised anomaly screening with a BiLSTM autoencoder: Isolation Forest and one-class SVM scores on learned latent activations are rank-averaged into a source-locked trial-level ensemble, evaluated with LOSO out-of-fold scoring against a screening pseudo-label (non-Healthy vs Healthy trials). Supplementary supervised pathology-tier models (XGBoost, LightGBM, random forest, SVM, MLP with nested RFECV LOSO) and a separate tabular one-class detector (isolation forest, LOF, and one-class SVM on engineered features) provide secondary benchmarks in `docs/paper/results.md`. Because evaluation uses cohort diagnostic membership rather than adjudicated incident falls, these results support gait **anomaly screening** as a non-invasive triage aid in mixed orthopedic–neurological populations, but **do not constitute direct fall prediction**; prospective validation with individual fall outcomes is required before clinical deployment.

---

## Metrics fill-in

_Auto-updated from pipeline artifacts — do not edit manually._

| Metric | Value |
|--------|-------|
| Primary BiLSTM-AE MCC (LOSO OOF) | 0.388 |
| Primary BiLSTM-AE AUROC (LOSO OOF) | 0.7545 |
| Primary BiLSTM-AE F1 | 0.786 |
| Primary BiLSTM-AE Sensitivity (Youden) | 0.711 |
| Primary BiLSTM-AE Specificity (Youden) | 0.720 |
| Primary BiLSTM-AE Cohen κ | 0.369 |
| Primary BiLSTM-AE PR-AUC | 0.8669 |
| Secondary deployable ensemble macro OvR AUC | 0.8404 (ensemble_soft_voting) |
| Best supervised single-model LOSO macro OvR AUC | 0.8415 (xgboost) |
| MCC abstract-lead threshold | 0.70 |

_Headline: report **AUROC** as threshold-independent headline; cite **MCC** and Cohen κ for rigor._

Regenerate after each pipeline run:

```bash
cd fall_risk_pipeline && python main.py
python ../scripts/regenerate_paper_results.py
```

