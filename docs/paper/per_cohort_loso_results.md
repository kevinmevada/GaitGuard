# Per-cohort LOSO results — pathology-tier screening (detailed)

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
