# Conclusion

> **RES-002:** Refresh metrics via `docs/paper/results.md` after each pipeline rerun.

## Summary

This work presents a reproducible end-to-end wearable IMU gait-screening pipeline for pathology-tier classification across eight clinical cohorts (260 participants; 1,356 trials). The pipeline combines supervised pathology-tier classification with a parallel unsupervised anomaly detection layer, providing both calibrated risk stratification and deviation-from-healthy flagging in a single inference call. Headline LOSO AUC values are reported in `docs/paper/results.md` (best single model vs primary deployable ensemble may differ — RES-003).

At the same time, results must be interpreted within study boundaries. Labels are cohort-level pathology categories rather than prospective participant-level fall outcomes; therefore, findings support pathology-tier gait screening research, not direct clinical fall prediction. Cross-cohort transfer instability and the lack of external prospective validation further reinforce this caution.

Overall, the pipeline provides a transparent and reproducible benchmark for multi-cohort IMU gait modeling and a strong basis for next-stage validation. Future work should prioritize prospective outcome-linked studies, external multi-site replication, and calibration against clinically used fall-risk instruments before deployment-oriented claims are made.
