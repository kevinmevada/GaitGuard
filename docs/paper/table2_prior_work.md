# Table 2 — Prior Work Comparison (Wearable Gait/Fall Literature vs This Study)

> **RES-002:** Update headline AUC from `docs/paper/results.md` after each pipeline rerun.

This table is split into two groups to prevent endpoint mixing. Only studies in **Table 2A** are acceptable for headline numeric comparison with pathology-tier/fall-risk screening. **Table 2B** is contextual only and must not be used for direct AUC ranking.

## Table 2A — Screening / Risk-Oriented Comparators (Directly Comparable Endpoints)

| Study | Year | Population (N) | Cohorts / Endpoint | Sensor setup | Best discriminative metric reported | Key limitation for direct comparison |
|---|---:|---:|---|---|---|---|
| Hausdorff et al. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/11494184/)) | 2001 | 52 | Community-dwelling older adults; prospective falls | Force-sensitive insoles | **No AUC reported** (stride-time variability predicted falls, logistic regression p<0.05) | Older single-cohort outpatient sample; non-IMU insole setup; no ROC-AUC table |
| Lockhart & Liu (closest match to requested 2008 nonlinear-dynamics comparator) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2892176/)) | 2008 | 13 | Fall-prone vs healthy adults during treadmill walking | Lab motion/kinematics (LDS, max Lyapunov exponent) | **No AUC reported** (group-discrimination significance) | Very small pilot sample; experimental induced-fall phenotype; not a multi-cohort clinical screening pipeline |
| Howcroft et al. ([JNER PDF](https://jneuroengrehab.biomedcentral.com/counter/pdf/10.1186/s12984-017-0255-9.pdf)) | 2017 | 100 | Older fallers vs non-fallers (retrospective 6-month fall history) | Insoles + tri-axial accelerometers (head/pelvis/shanks) | **No AUC reported** (best model accuracy 78%) | Retrospective binary endpoint; different sensors and label definition vs pathology-tier multiclass screening |
| Mirelman-associated PD fall prediction model ([Movement Disorders abstract](https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.25404)) | 2013 | 205 | Parkinson's disease; future falls (6-month) | Clinical + gait/sway predictors | AUC **0.83** (full model), **0.80** (3-test simplified model) | Single-disease PD cohort; not multi-cohort orthopedic+neurological+healthy discrimination |
| **This work (GaitGuard)** — metrics from `docs/paper/results.md` | 2026 | 260 | 8 cohorts; pathology-tier multiclass screening | 4 IMUs; sensor ablation reported | Best LOSO AUC + primary ensemble AUC: _regenerate after pipeline rerun_ | Internal LOSO on one public dataset; no prospective fall-outcome validation |

## Table 2B — Contextual Literature (Non-Comparable Endpoints; Do Not Rank Numerically)

| Study | Year | Population (N) | Endpoint class | Reported metric | Why non-comparable to Table 2A |
|---|---:|---:|---|---|---|
| Weiss et al. (lower-back inertial benchmark) ([Sensors 2020 PDF](https://mdpi-res.com/d_attachment/sensors/sensors-20-06479/article_deploy/sensors-20-06479.pdf?version=1605246263)) | 2020 | 143 real-world falls + ADL windows | **Fall event detection** (window/event-level), not subject-level risk screening | AUC up to **0.996** | Different task definition, label granularity, and evaluation target; this value must not be compared directly with pathology-tier/risk-screening AUCs |
| Tao/Zhao et al. Sensors review ([Sensors 2022](https://www.mdpi.com/1424-8220/22/18/6752)) | 2022 | 25 included studies | Systematic review (mixed tasks/endpoints) | Reported ranges (accuracy 0.57–0.90; many studies reported ROC/AUC) | Review paper, heterogeneous endpoints, no single benchmark model |

## Notes for manuscript text

- Use only **Table 2A** studies for direct numeric comparison in Results/Discussion text.
- Keep **Table 2B** strictly contextual; do not compute ranking deltas against those metrics.
- Explicitly state that Weiss 2020 is event-detection (not risk screening) whenever cited.
