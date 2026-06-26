# Per-cohort LOSO results — pathology-tier screening (detailed)

This section reports **cohort-resolved** LOSO out-of-fold performance. Pooled means are supplementary only; clinical heterogeneity across the eight Voisard cohorts is the primary result.

**Model:** bilstm_ae_ensemble (`bilstm_ae_ensemble_score`)  
**Global Youden threshold (all pathological vs Healthy):** 0.2937

> Voisard does not include an MS-labelled cohort. Neuropathy-tier signal is carried by **CIPN** (chemotherapy-induced peripheral neuropathy) and **RIL** (radiculopathy/leg pain). Orthopedic cohorts map to manuscript aliases **HOA** (HipOA) and **TKA** (KneeOA).

## 1. One-vs-Healthy screening per pathological cohort

Each row is a **separate** binary task: cohort *c* (positive) vs Healthy (negative). AUROC and F1 are **not** macro-averaged across cohorts.

| Cohort | vs Healthy | n trials (path.) | n participants | AUROC | F1 | MCC | Sens. | Spec. | Anomaly rate (%) | Ref. fall prob. (%) | Mean score gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PD** | PD vs Healthy | 6 | 2 | 0.8958 | 0.8333 | 0.7083 | 0.8333 | 0.8750 | 83.3 | 67.3 | 0.1039 |
| **CVA** | CVA vs Healthy | 6 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 54.2 | 0.4534 |
| **CIPN** | CIPN vs Healthy | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 41.8 | 0.5725 |
| **RIL** | RIL vs Healthy | 4 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 38.9 | 0.5434 |
| **HOA** | HOA vs Healthy | 6 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 28.5 | 0.4475 |
| **TKA** | TKA vs Healthy | 6 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 24.1 | 0.3602 |
| **ACL** | ACL vs Healthy | 6 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0 | 18.7 | 0.3741 |

### Interpretation guide

- **AUROC / F1** — discrimination for that cohort only; compare across rows, do not average.
- **Anomaly rate** — % of pathological trials flagged at the cohort-specific Youden threshold (re-fit on Healthy + that cohort's OOF trials).
- **Ref. fall prob.** — literature reference fall-risk percentage for the cohort label (not a prospective outcome in this dataset).
- **Mean score gap** — pathological minus healthy mean anomaly score on the same comparison set.

## 2. Anomaly score distribution by cohort (all eight cohorts)

| Cohort | n trials | n participants | Mean score | Median | SD | Anomaly rate (%) | Ref. fall prob. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Healthy | 8 | 2 | 0.1782 | 0.1695 | 0.0445 | 0.0 | 5.2 |
| **PD** | 6 | 2 | 0.2821 | 0.2969 | 0.0621 | 66.7 | 67.3 |
| CVA | 6 | 2 | 0.6315 | 0.6156 | 0.0604 | 100.0 | 54.2 |
| CIPN | 4 | 2 | 0.7506 | 0.7911 | 0.0905 | 100.0 | 41.8 |
| RIL | 4 | 2 | 0.7216 | 0.7218 | 0.0492 | 100.0 | 38.9 |
| HOA | 6 | 2 | 0.6256 | 0.6178 | 0.0533 | 100.0 | 28.5 |
| TKA | 6 | 2 | 0.5384 | 0.5320 | 0.0342 | 100.0 | 24.1 |
| ACL | 6 | 2 | 0.5522 | 0.5534 | 0.0315 | 100.0 | 18.7 |

## 3. Kruskal-Wallis — cohort differences in anomaly score

Tests whether anomaly-score distributions differ across cohorts (non-parametric; trial-level and participant-mean variants).

- **Trial-level scores:** H = 39.351, p = 0.0000 (significant at α=0.05), k = 8 cohorts
- **Participant-mean scores:** H = 14.118, p = 0.0491 (significant at α=0.05), k = 8 cohorts

## 4. Clinical discussion — do not average away cohort signal

**PD clinical paradox (discuss explicitly):** Parkinson's disease carries the highest reference fall probability in this dataset (67.3%), yet LOSO anomaly flagging is comparatively low (83.3% of PD trials above the Youden threshold). This pattern is clinically meaningful — PD gait can be pathologically impaired yet **internally consistent** (narrow, stereotyped kinematics), producing modest reconstruction/latent deviation from a healthy manifold. Averaging PD into a single pooled metric would hide this dissociation between epidemiological fall risk and unsupervised anomaly score.

The eight-cohort Voisard design enables contrasts that single-disease studies cannot replicate: high fall-probability neurological cohorts (PD, CVA) vs orthopedic mechanical gait (HOA, TKA, ACL) vs neuropathy-tier cohorts (CIPN, RIL). Report each row in the main text; reserve pooled AUROC for supplementary material only.
