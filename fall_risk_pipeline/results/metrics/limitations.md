# Prediction export — limitations

**Research prototype — not for clinical use without validation.**

- Research prototype — not for clinical use without validation.
- Retrospective secondary analysis of existing recordings — no forward-in-time data collection for this project.
- Single open dataset (Voisard et al., Figshare 10.6084/m9.figshare.28806086); no external multi-site replication reported.
- No prospective participant follow-up or incident-outcome ascertainment for this analysis.
- No participant-level ground-truth fall outcomes; labels are cohort-level pathology tiers and literature-based fall-risk references only.
- Internal LOSO metrics on the same dataset — not independent prospective performance.
- Outputs are exploratory screening scores from open IMU gait data, not medical advice or a diagnosis.
- Single-trial API inference does not replicate multi-trial patient aggregation used in training.
- Risk thresholds (Youden J) are derived from internal cross-validation, not regulatory clearance.
- Do not use these results for treatment or fall-prevention decisions without prospective validation.
