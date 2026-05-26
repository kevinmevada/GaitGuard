# Paper draft — title and abstract

**Journal target:** *Sensors* (MDPI)  
**Status:** Abstract structure complete; **primary AUC and 95% CI must be inserted after** `python main.py --stage evaluate` (see [Metrics fill-in](#metrics-fill-in)).

---

## Working title

**Multi-Sensor Wearable IMU Gait Features for Subject-Independent Pathology-Tier Screening: A Leave-One-Subject-Out Study in 260 Clinical Participants**

*Alternates:*  
- *Pathology-Stratified Fall-Risk Screening from Clinical Wearable Gait IMU: 1,356 Trials, Eight Cohorts*  
- *GaitGuard: Ensemble Machine Learning on Open Multi-Cohort Wearable Gait Data with Nested Leave-One-Subject-Out Validation*

---

## Abstract (five sentences — *Sensors* structure)

Falls affect approximately one in three community-dwelling adults aged ≥65 years annually and remain a leading cause of injury-related morbidity, hospitalization, and loss of independence worldwide. Wearable inertial measurement unit (IMU) gait analysis can quantify mobility impairment in clinical and ambulatory settings, yet many methods report trial-level classifiers without leave-one-subject-out (LOSO) validation, omit multi-trial session aggregation (mean, variability, and trajectory across repeated walks), or merge heterogeneous orthopedic and neurological pathologies into a single “high-risk” label, limiting both generalizability and interpretability for fall-prevention workflows. We developed and evaluated a reproducible pipeline on an open eight-cohort clinical dataset (N = 260 participants, 1,356 walking trials, four body-worn IMUs at 100 Hz) that extracts temporal, spectral, trunk-dynamics, orientation, and asymmetry features; aggregates them to participant-level statistics (mean, standard deviation, range, and linear trend across within-session trials); selects ≤20 predictors via grouped recursive feature elimination; and trains an XGBoost–LightGBM–random forest–support vector machine soft-voting ensemble under nested LOSO cross-validation with three pathology-tier labels (healthy, orthopedic elevated risk, neurological elevated risk). In LOSO evaluation, the primary soft-voting ensemble achieved macro one-vs-rest area under the receiver operating characteristic curve (AUC) of **X.XX** (95% bootstrap confidence interval **X.XX–X.XX**) for three-class discrimination, with companion unsupervised anomaly detection (isolation forest, local outlier factor, one-class SVM) flagging gait patterns deviating from healthy training references. These results support multi-sensor wearable gait IMU as a non-invasive stratification aid to prioritize referral and longitudinal monitoring in mixed orthopedic–neurological populations, while remaining screening-oriented rather than a standalone diagnostic for individual fall prediction.

---

## Metrics fill-in

Replace **`X.XX`** in sentence 4 with the **ensemble** (or `ensemble_soft_voting`) row from:

`fall_risk_pipeline/results/metrics/metrics.csv`

| Column | Abstract use |
|--------|----------------|
| `auc` | Macro one-vs-rest AUC → first **X.XX** |
| `auc_ci_low` | Lower bound of 95% CI → second **X.XX** |
| `auc_ci_high` | Upper bound of 95% CI → third **X.XX** |

After a full pipeline re-run (`features` → `select_features` → `train` → `evaluate` → `report`), extract values with:

```bash
cd fall_risk_pipeline
python -c "
import pandas as pd
m = pd.read_csv('results/metrics/metrics.csv')
row = m[m['model'].str.contains('ensemble', case=False, na=False)].iloc[0]
print(f\"AUC={row['auc']:.2f}, 95% CI [{row['auc_ci_low']:.2f}, {row['auc_ci_high']:.2f}]\")
"
```

**Note:** Stale checkpoints or pre–multiclass / pre–aggregation artifacts produce non-publishable numbers; regenerate metrics on the current `main` branch before submission.

Optional sensitivity sentence (binary neurological high-risk only, `label_mode: binary`, `binary_strategy: threshold_ge_2`): report a second AUC from the same table if reviewers request a dichotomous endpoint.

---

## Full paper outline (*Sensors* — quantified sections to draft next)

| Section | Content to quantify |
|---------|---------------------|
| **Introduction / background** | Fall epidemiology; IMU gait in aging and neuro/orthopedic disease; cite dataset (Figshare 10.6084/m9.figshare.28806086). |
| **Gap** | Subject leakage, trial vs patient features, label heterogeneity, lack of open reproducible LOSO benchmarks on this cohort. |
| **Methods** | N = 260, 1,356 trials, 8 cohorts, 4 IMUs, 100 Hz; preprocessing (Madgwick, gait events); feature families; RFECV cap ≤20; nested LOSO; Optuna trials; ensemble; macro-OVR AUC, macro F1, per-class metrics; **Youden J cutoff** (sens/spec at primary threshold); Morse/STRATIFY cited for screening context; DeLong/bootstrap comparisons. |
| **Results** | Table 1 demographics; AUC **X.XX (95% CI …)**; calibration; SHAP top features; **feature ablation** (`feature_ablation.csv`: all features, top-10 SHAP, leave-one-group-out, minus Lyapunov); anomaly vote rates by cohort; gait-event validation error (ms). |
| **Discussion / conclusion** | Screening vs diagnosis; single-trial API limitation; **limitations** (research prototype, no clinical decision support). |
| **Limitations** | **Retrospective**, **single dataset**, **no prospective follow-up**, **no ground-truth fall outcomes** (cohort labels only); internal LOSO only; path to prospective validation — `docs/limitations.md`, `docs/paper/limitations.md`. |
| **Ethics** | Public de-identified Figshare data; CPP Île-de-France II (2014-10-04 RNI); no new human data — see `docs/paper/ethics_statement.md`. |

See also: `docs/label_binning.md`, `docs/inference_single_trial_limitation.md`, `docs/ethics.md`, `fall_risk_pipeline/docs/reproducibility.md`.
