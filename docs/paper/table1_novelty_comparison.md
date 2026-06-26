# Table 1 — Methodological novelty vs competitor literature (Section 2)

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
