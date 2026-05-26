# Limitations and non-clinical use

GaitGuard is a **research prototype** for analyzing public de-identified gait IMU data. It is **not** a medical device and **not** cleared for clinical decision-making.

## Disclaimer

**Research prototype — not for clinical use without validation.**

---

## Study design (key limitations for *Sensors*)

This work is a **retrospective, secondary analysis** of a **single public dataset**. The following limits must be stated explicitly in any clinical-facing manuscript:

| Limitation | What we did / did not do |
|------------|-------------------------|
| **Retrospective study** | All model development and evaluation used **existing** recordings and metadata. No protocol was run forward in time for this project. |
| **Single dataset** | One open cohort (Voisard et al., Figshare [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086); N = 260 participants, 1,356 trials). **No external multi-site replication** is reported here. |
| **No prospective follow-up** | Participants were **not** followed after gait recording to collect falls, injurious falls, or healthcare utilization for this analysis. |
| **No ground-truth fall outcomes** | Labels are **cohort-level pathology tiers** and literature-based fall-risk references per cohort—not verified individual fall counts or incident falls during a defined observation window. |

These constraints are **acceptable for a methods-focused *Sensors* paper** when stated clearly; they **do not** support claims of clinical effectiveness or fall prediction in deployment.

---

## Additional technical limitations

1. **Internal validation only** — Performance metrics (e.g. LOSO macro AUC, Youden sensitivity/specificity) are from **cross-validation on the same dataset**, not independent prospective testing.

2. **Not medical advice** — Text such as “may warrant further clinical assessment” is intentionally non-directive. It does **not** recommend a specific treatment, admission, or intervention.

3. **Label heterogeneity** — Binary collapse of orthopedic and neurological tiers (if used) merges clinically distinct groups; multiclass pathology tiers are preferred for reporting.

4. **Single-trial API inference** — The public API scores one uploaded walk. Training used multi-trial patient aggregation (mean, variability, trend). See [`inference_single_trial_limitation.md`](inference_single_trial_limitation.md).

5. **Thresholds** — The operating point uses Youden’s J from internal LOSO validation. It is **not** calibrated to Morse Fall Scale, STRATIFY, or other bedside tools without a dedicated study.

6. **Generalization** — Sensor type, placement, walking protocol, and cohort mix may not match other populations. External validation is required before operational use.

7. **Anomaly detection** — Unsupervised flags mark deviation from healthy training references; they are not validated predictors of falls or adverse events.

---

## Path to prospective validation (future work)

To move beyond retrospective cohort classification, we recommend:

1. **Prospective cohort** — Enroll a new sample with pre-specified inclusion criteria, device protocol, and follow-up duration (e.g. 6–12 months).

2. **Adjudicated fall outcomes** — Record **incident falls** (and injurious falls) via diaries, clinician report, or electronic health records; use time-to-first-fall or fall rate as **primary endpoints**, not cohort labels alone.

3. **External validation** — Apply frozen models (or retrain with locked feature schema) at **independent site(s)** with blinding of outcome ascertainment where feasible.

4. **Calibration and decision-curve analysis** — Relate model scores to observed fall risk; compare against Morse Fall Scale / STRATIFY and clinician judgment.

5. **Regulatory and governance review** — Institutional approval for **new** data collection; distinguish secondary public-data analysis from prospective interventional monitoring.

---

## Appropriate use

- Methods research, reproducible benchmarks, and teaching
- Hypothesis generation for future prospective studies

## Inappropriate use

- Sole basis for diagnosis, triage, treatment, or fall-risk disposition
- Claiming individual fall prediction without incident fall follow-up
- Replacing clinician judgment or established fall-risk assessments

---

## Manuscript (*Sensors*)

Use the **Study design** table and **Path to prospective validation** subsections in the paper **Limitations** section. Pair with ethics in [`paper/ethics_statement.md`](paper/ethics_statement.md).

Suggested one-paragraph opener for the paper:

> This was a **retrospective** machine-learning study on a **single** publicly available gait dataset. We did **not** perform **prospective follow-up** and had **no participant-level ground-truth fall outcomes**; supervision used **cohort-level pathology labels** and published fall-risk references only. Findings support feasibility of IMU-based screening research but require **prospective validation** with adjudicated falls before clinical use.
