# Conclusion

This work presents a reproducible end-to-end wearable IMU gait-screening pipeline for pathology-tier classification across eight clinical cohorts (260 participants; 1,356 trials). Using participant-grouped evaluation, the tabular branch achieved strong multiclass discrimination (best macro one-vs-rest AUC 0.916 for random forest), while deep architectures showed competitive LOSO macro-F1 and accuracy.

A key practical result is that reduced sensor configurations can retain high performance: in sensor-ablation analysis, a two-sensor setup (head + right foot) reached AUC 0.9336, exceeding the full four-sensor configuration (AUC 0.9273). Explainability analyses consistently highlighted trunk-dynamics variability features, supporting a biomechanically plausible interpretation rather than purely opaque model behavior.

At the same time, results must be interpreted within study boundaries. Labels are cohort-level pathology categories rather than prospective participant-level fall outcomes; therefore, findings support pathology-tier gait screening research, not direct clinical fall prediction. Cross-cohort transfer instability and the lack of external prospective validation further reinforce this caution.

Overall, the pipeline provides a transparent and reproducible benchmark for multi-cohort IMU gait modeling and a strong basis for next-stage validation. Future work should prioritize prospective outcome-linked studies, external multi-site replication, and calibration against clinically used fall-risk instruments before deployment-oriented claims are made.
