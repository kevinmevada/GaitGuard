# Conclusion

## Summary

We presented a healthy-reference gait anomaly screening system evaluated under strict leave-one-subject-out validation across eight clinical cohorts (260 participants; 1,355 trials), achieving ROC-AUC 0.7545 without pathological training labels. Isolation Forest on BiLSTM-AE latents is the effective in-domain detector (0.7540); the source-locked 2-method ensemble is reported for protocol consistency. A source-locked zero-shot cross-dataset transfer evaluation on DAPHNET revealed a fusion-behavior reversal under domain shift (reconstruction AUC 0.7046 vs. ensemble 0.5314), disclosed transparently alongside its single-positive-subject (S03) case-study limitation.

We additionally identified and corrected a data-contamination issue in a secondary supervised tabular pipeline (9 DAPHNET subjects leaked into patient-level features), reporting affected results as provisional pending a corrected Voisard-only (N=260) re-run. Deep supervised baselines and the primary BiLSTM-AE anomaly path use clean N=260 loaders and remain the reliable numeric anchors.

Labels are cohort-level pathology categories rather than prospective participant-level fall outcomes; findings support pathology-tier gait screening research, not direct clinical fall prediction. We believe the combination of methodological transparency — including disclosure of confounds and errors discovered during development — and cross-dataset evaluation offers a template for more rigorous, reproducible wearable gait screening research.

## Future work

1. Complete the Voisard-only tabular recompute (N=260) and refresh provisional supervised / ablation / LOCO numbers.
2. Prospective outcome-linked studies with adjudicated incident falls.
3. External multi-site replication with frozen models.
4. FOG transfer evaluation with multiple positive-bearing subjects.
5. Calibration against clinically used fall-risk instruments before deployment-oriented claims.

## Reproducibility

Code, LOSO split manifests, and analysis artifacts are maintained in this repository; Zenodo DOI and Hugging Face Hub checkpoint URLs should be inserted in the Data Availability Statement once minted.
