# Paper draft — title and abstract

**Journal target:** *Sensors* (MDPI)  
**Status:** Abstract complete with evaluated metrics (AUC 0.91, 95% CI 0.88–0.94).

---

## Working title

**End-to-End Deep Learning vs Handcrafted Features for Wearable IMU Fall-Risk Screening Across Eight Clinical Cohorts: A Leave-One-Subject-Out Comparison in 260 Participants**

*Alternates:*  
- *Deep Learning and Ensemble Machine Learning for Multi-Sensor IMU Gait Screening: 1,356 Trials, Eight Cohorts, LOSO Validation*  
- *GaitGuard: Comparing InceptionTime, Transformer, and Handcrafted Ensemble Pipelines on Open Multi-Cohort Wearable Gait Data*

---

## Abstract (five sentences — *Sensors* structure)

Falls affect approximately one in three community-dwelling adults aged ≥65 years annually and remain a leading cause of injury-related morbidity, hospitalization, and loss of independence worldwide. Wearable inertial measurement unit (IMU) gait analysis can quantify mobility impairment in clinical and ambulatory settings, yet many methods report trial-level classifiers without leave-one-subject-out (LOSO) validation, omit multi-trial session aggregation (mean, variability, and trajectory across repeated walks), or merge heterogeneous orthopedic and neurological pathologies into a single “high-risk” label, limiting both generalizability and interpretability. We developed and evaluated a reproducible pipeline on an open eight-cohort clinical dataset (N = 260 participants, 1,356 walking trials, four body-worn IMUs at 100 Hz) that extracts temporal, spectral, trunk-dynamics, orientation, and asymmetry features; aggregates them to participant-level statistics (mean, standard deviation, range, and linear trend across within-session trials); selects ≤20 predictors via grouped recursive feature elimination; and trains an XGBoost–LightGBM–random forest–support vector machine soft-voting ensemble under nested LOSO cross-validation to discriminate three **pathology-tier** labels (healthy, orthopedic, neurological) as **proxies for fall-risk stratification**; importantly, the dataset provides cohort-level diagnostic labels, not individual fall-outcome ground truth. In LOSO evaluation, the primary soft-voting ensemble achieved macro one-vs-rest area under the receiver operating characteristic curve (AUC) of **0.91** (95% bootstrap confidence interval **0.88–0.94**) for three-class pathology-tier discrimination, with companion unsupervised anomaly detection (isolation forest, local outlier factor, one-class SVM) flagging gait patterns deviating from healthy training references. Because supervision is based on diagnostic cohort membership rather than adjudicated incident falls, these results support pathology-tier gait screening as a non-invasive stratification aid for prioritizing referral and longitudinal monitoring in mixed orthopedic–neurological populations, but **do not constitute direct fall prediction**; prospective validation with individual fall outcomes is required before clinical deployment.

---

## Metrics fill-in

Metrics populated from `fall_risk_pipeline/results/metrics/metrics.csv` (ensemble_soft_voting row):

| Metric | Value |
|--------|-------|
| Macro OvR AUC | 0.91 |
| 95% bootstrap CI lower | 0.88 |
| 95% bootstrap CI upper | 0.94 |

To regenerate after a pipeline re-run:

```bash
cd fall_risk_pipeline
python -c "
import pandas as pd
m = pd.read_csv('results/metrics/metrics.csv')
row = m[m['model'].str.contains('ensemble', case=False, na=False)].iloc[0]
print(f\"AUC={row['auc']:.2f}, 95% CI [{row['auc_ci_low']:.2f}, {row['auc_ci_high']:.2f}]\")
"
```

Optional sensitivity sentence (binary neurological high-risk only, `label_mode: binary`, `binary_strategy: threshold_ge_2`): report a second AUC from the same table if reviewers request a dichotomous endpoint.

---

## Full paper outline (*Sensors* — quantified sections to draft next)

| Section | Content to quantify |
|---------|---------------------|
| **Introduction / background** | Fall epidemiology; IMU gait in aging and neuro/orthopedic disease; cite dataset (Figshare 10.6084/m9.figshare.28806086). |
| **Gap** | Subject leakage, trial vs patient features, label heterogeneity, lack of open reproducible LOSO benchmarks on this cohort. |
| **Methods** | N = 260, 1,356 trials, 8 cohorts, 4 IMUs, 100 Hz; preprocessing (Madgwick, gait events); feature families; RFECV cap ≤20; nested LOSO; Optuna trials; ensemble; macro-OVR AUC, macro F1, per-class metrics; **Youden J cutoff** (sens/spec at primary threshold); Morse/STRATIFY cited for screening context; DeLong/bootstrap comparisons. |
| **Results** | Table 1 demographics; AUC **0.91 (95% CI 0.88–0.94)**; calibration; SHAP top features; **feature ablation** (`feature_ablation.csv`: all features, top-10 SHAP, leave-one-group-out, minus Lyapunov); anomaly vote rates by cohort; gait-event validation error (ms). |
| **Discussion / conclusion** | Screening vs diagnosis; single-trial API limitation; **limitations** (research prototype, no clinical decision support). |
| **Limitations** | **Retrospective**, **single dataset**, **no prospective follow-up**, **no ground-truth fall outcomes** (cohort labels only); internal LOSO only; path to prospective validation — `docs/limitations.md`, `docs/paper/limitations.md`. |
| **Ethics** | Public de-identified Figshare data; CPP Île-de-France II (2014-10-04 RNI); no new human data — see `docs/paper/ethics_statement.md`. |

See also: `docs/label_binning.md`, `docs/inference_single_trial_limitation.md`, `docs/ethics.md`, `fall_risk_pipeline/docs/reproducibility.md`.
