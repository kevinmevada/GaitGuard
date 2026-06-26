# Limitations section — manuscript draft (*Sensors*)

Copy into the paper **Limitations** (or end of **Discussion**). Adjust tense/voice to match the rest of the manuscript.

---

## Limitations

This study has several important limitations. First, it was a **retrospective** analysis of **one** open clinical gait dataset (260 participants, 1,356 trials; Voisard et al., Figshare DOI: 10.6084/m9.figshare.28806086). We performed **no prospective enrollment or follow-up** for this work and therefore report **no incident fall outcomes** (falls, injurious falls, or fall-related healthcare use) linked to individual participants. Supervision relied on **cohort-level pathology labels** and literature-based fall-risk references by diagnostic group—not verified per-participant fall histories during a defined observation window. Consequently, model performance (e.g. leave-one-subject-out macro AUC and Youden operating-point sensitivity and specificity) reflects **cohort discrimination and internal cross-validation**, not prospective fall prediction or clinical effectiveness.

Second, evaluation was **entirely on the same dataset** used for feature and model development. We did not include external multi-site replication. Single-trial deployment inference (API) does not reproduce multi-trial patient-level aggregation (mean, standard deviation, range, and trend) used in training, which may underestimate uncertainty for real-world monitoring.

Feature selection during evaluation uses **nested leave-one-subject-out RFECV** on each training fold (`nested_in_evaluation: true`), so reported accuracy and AUC are not inflated by global feature screening; global RFECV still defines the deployment feature list in `selected_features.json`.

Third, this system is a **research prototype**. Outputs are not medical advice, regulatory-cleared diagnostics, or substitutes for established tools such as the Morse Fall Scale or STRATIFY. Risk thresholds were derived from internal Youden J optimization and are not calibrated to bedside screening scores without further study.

Fourth, **laterality imbalance (sidestep by design).** Several orthopedic and neurological cohorts in the Figshare corpus encode an affected-limb side in trial `meta.json` (`laterality`). At the participant level we observe: **Hip OA** — 15 right / 0 left / 0 bilateral; **CVA** — 47 right / 2 left / 0 bilateral; **ACL** — 9 right / 2 left (11 participants; no published reference split). The acquisition protocol uses a fixed walkway with a mandated U-turn; when most deficits are right-labeled, straight and return bouts can couple **path side** with **affected limb**, so gait asymmetry features may partly reflect protocol geometry rather than isolated pathology. We flag Hip OA and CVA as `laterality_biased` in pipeline metadata and report side counts in `results/metrics/laterality_audit.csv` (regenerate via `python scripts/audit_laterality.py`). Prior wearable-gait screening studies (e.g. Moon et al., 2020; Trabassi et al., 2022) do not stratify or report laterality confounds; **explicit documentation here is intentional** and should be read as a methodological transparency choice, not a claim of superiority. Analyses that pool left- and right-affected participants without side stratification should be interpreted cautiously.

**Note:** `clinicalDeficitSide` in the same metadata files does not always agree with `laterality` (e.g. 5/15 Hip OA participants); all tabulated side counts in the audit use **`laterality`** to match the dataset’s published bookkeeping.

### Future directions

**Prospective validation** should enroll a new cohort, record **adjudicated fall outcomes** over a pre-specified follow-up period, and test frozen or externally trained models at independent site(s), with calibration and comparison to clinical fall-risk assessments. Until such evidence exists, results should be interpreted as supporting IMU-based gait screening **research**, not routine clinical decision-making.

---

## Short bullet list (for cover letter or reviewer response)

- Retrospective; single public dataset  
- No prospective follow-up  
- No ground-truth individual fall outcomes  
- Cohort-level / pathology-tier labels only  
- LOSO internal validation; no external replication  
- Research prototype — not for clinical use without validation  
- **Laterality skew (HOA 15R/0L; CVA 47R/2L) — sidestep protocol confound; see `laterality_audit.csv`**
