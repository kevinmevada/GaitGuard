# Discussion

## 1. Principal findings

This study evaluated a reproducible wearable-IMU gait screening pipeline for pathology-tier classification across eight clinical cohorts. Under participant-grouped evaluation, tabular models achieved strong multiclass discrimination, with random forest reaching macro one-vs-rest AUC 0.916 and ensemble methods reaching AUC 0.911. Class-wise analysis showed the highest separability for neurological participants and comparatively lower performance for the orthopedic tier, consistent with the expected overlap between orthopedic gait adaptations and age-related variability.

## 2. Interpretation of model behavior

The explainability analyses indicate that discrimination is driven primarily by lower-back dynamic variability and range-related descriptors (e.g., AP range dispersion and ML jerk-derived features), with supportive contributions from head-to-trunk stability ratios. The top feature, `lb_range_ap_std`, can be interpreted clinically as the walk-to-walk inconsistency of trunk-level forward acceleration amplitude rather than as an opaque model token: higher values imply less repeatable propulsion/braking control across repeated trials. This pattern is biomechanically plausible because unstable trunk control and variable AP excursion are expected in neurologic and mixed high-risk gait phenotypes. Cohort-level SHAP summaries further suggest that the same feature families generalize across diagnostic groups, while effect magnitude differs by cohort composition.

## 3. Relation between classical and deep approaches

In this run, deep models demonstrated competitive LOSO macro-F1/accuracy, with TCN performing strongest among deep architectures. However, exported deep-model AUC fields were not populated in the current metrics artifact, which limits direct AUC-ranked comparison against the tabular branch in this manuscript draft. Even with that constraint, the results support a practical conclusion: both representation strategies carry useful signal, and the tabular branch currently provides the clearest calibrated reporting path for publication-grade primary endpoints.

## 4. Sensor configuration implications

Sensor-ablation findings suggest that high screening performance can be preserved with reduced hardware. The strongest practical result was that a two-sensor setup (head + right foot) achieved AUC 0.9336, outperforming the full four-sensor setup (AUC 0.9273). Two-sensor and even single-sensor setups therefore remained competitive relative to full instrumentation, which is important for deployment feasibility, participant burden, and acquisition cost. In particular, head-containing subsets and lower-back-only configurations retained most discriminative power, whereas foot-only combinations performed notably worse. This supports tiered deployment designs in which richer sensor sets are optional rather than mandatory.

## 5. Cross-cohort transfer and external validity signal

Leave-one-cohort-out transfer results highlight a central translational challenge: when the held-out cohort contains a single label class, AUC becomes undefined and conventional discrimination summaries are not informative. The fallback confidence metrics (mean true-class probability) reveal strong heterogeneity in transferability across cohorts, indicating that model certainty degrades when pathology distributions shift. This is an expected but important finding, reinforcing that internal LOSO robustness does not guarantee cross-cohort transport without targeted external adaptation or calibration.

## 5.1 Leakage-sensitivity interpretation

The grouped-vs-ungrouped comparison showed only small AUC differences in this dataset (maximum inflation +2.33% for LightGBM, near-zero average change across models, and negative deltas for random forest and MLP). This pattern is reassuring and supports the stability of reported grouped results, but it should not be overinterpreted as evidence of a large leakage artifact in this specific cohort composition.

## 6. Reproducibility and reporting contribution

A practical contribution of this work is the explicit reproducibility pathway: stage-structured execution, configuration-controlled preprocessing/modeling, deterministic seed controls, and one-command containerized reruns. For clinical-ML literature, these engineering details are not ancillary; they directly affect trust, auditability, and independent verification. By pairing performance reporting with artifact-level reproducibility, the pipeline addresses common barriers that prevent published IMU models from being replicated.

## 7. Clinical interpretation boundary

These findings support pathology-tier gait screening as a research-oriented stratification aid, not direct prospective fall prediction. Labels in this dataset represent cohort-level diagnostic categories rather than participant-level adjudicated future falls. Therefore, the reported metrics should be interpreted as internal evidence of discriminative screening capacity within this dataset context, and not as proof of bedside clinical efficacy.

## 8. Limitations and future directions

Key limitations remain: retrospective single-dataset design, no prospective fall-outcome endpoint, no external multi-site validation, and single-trial API inference that does not fully reproduce multi-trial participant aggregation used during training. Future work should prioritize (i) prospective validation with incident fall outcomes, (ii) external cohort replication with frozen models, (iii) cohort-shift-aware calibration, and (iv) harmonized reporting of deep and tabular metrics on identical primary endpoints.

## 9. Conclusion

In summary, the presented pipeline demonstrates that participant-grouped, reproducible wearable-IMU modeling can achieve strong multiclass pathology-tier discrimination in a heterogeneous open clinical dataset. The combination of robust tabular performance, interpretable feature attribution, and sensor-efficiency evidence provides a credible foundation for next-stage prospective validation studies.
