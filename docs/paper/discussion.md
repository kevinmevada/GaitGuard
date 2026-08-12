# Discussion

> **PUB-002:** Do not cite run-specific numbers until `python main.py` + `scripts/regenerate_paper_results.py`. All quantitative claims below must be rewritten from regenerated artifacts (`metrics.csv`, SHAP exports, `sensor_ablation.csv`, `split_protocol_comparison.csv`, `deep_learning_metrics.csv`).

## 1. Principal findings

This study evaluated a reproducible wearable-IMU gait **screening** pipeline for **pathology-tier** classification across eight clinical cohorts — a proxy stratification task, not adjudicated incident-fall prediction (RES-001). Under participant-grouped LOSO evaluation with nested per-fold RFECV, tabular models are expected to show tier-dependent separability; headline macro OvR AUC values will be reported in `docs/paper/results.md` after regeneration (best single nested-Loso model vs primary deployable ensemble may differ — ML-032 / RES-003). Qualitative tier patterns (e.g., neurological vs orthopedic difficulty) should be confirmed from per-class metrics in the regenerated `metrics.csv`, not assumed here.

## 2. Interpretation of model behavior

Explainability results will be drawn from regenerated LOSO SHAP exports after the pipeline rerun. Interpretation should focus on biomechanically plausible families (trunk dynamics, variability/range descriptors, head–trunk stability ratios) rather than citing specific feature names until `feature_ablation.md` and SHAP tables confirm rankings. Cohort-level SHAP summaries should be used to assess whether effect directions are consistent across diagnostic groups or driven by cohort composition.

## 3. Relation between classical and deep approaches

Deep and tabular branches use aligned LOSO participant holdout. Both re-tune on each outer train fold: tabular via full Optuna search; deep via per-fold learning-rate Optuna on the inner participant validation split (`loso_hyperparameter_tuning.enabled: true`, ML-042 / HIGH-001). After regeneration, compare macro OvR AUC from `deep_learning_metrics.csv` against nested tabular rows in `metrics.csv` (filtered by `feature_selection_protocol`). Do not cite a specific deep architecture (e.g., TCN) as “strongest” until the regenerated DL table supports it.

## 4. Sensor configuration implications

Sensor-efficiency claims must come from regenerated `sensor_ablation.csv` (LOSO + nested RFECV intersect). Until then, state only that the pipeline **evaluates** all sensor subsets and exports rankings to `docs/paper/results.md` — do not assert which subsets retain “most” discriminative power.

## 5. Cross-cohort transfer and external validity signal

Leave-one-cohort-out (LOCO) transfer results highlight translational limits: held-out cohorts with single-class test composition yield undefined AUC; small train cohorts are flagged `unstable_small_n` (ML-044/ML-051). Regenerated `cross_cohort_transfer.csv` should be used for external-validity wording. Internal LOSO robustness does not imply cross-cohort transport without adaptation or calibration.

## 5.1 Split-protocol sensitivity (ML-048)

The grouped LOSO vs StratifiedKFold comparison (`split_protocol_comparison.csv`) estimates how much performance changes when the validation split is easier (more training subjects per fold), not duplicate-subject leakage at the participant matrix. After regeneration, report mean and maximum inflation from that table; do not cite historical percentages from prior manuscript drafts.

### In-domain vs. zero-shot fusion behavior

Under strict LOSO on Voisard, the source-locked 2-method ensemble (Isolation Forest + One-Class SVM on
BiLSTM-AE latents, rank-averaged) slightly exceeds Isolation Forest alone (AUC 0.7545 vs. 0.7540; +0.0005) and substantially exceeds AE reconstruction error (0.4790). The ensemble is reported as the protocol primary endpoint but is empirically IF-dominated; OCSVM contributes negligible in-domain signal. Under zero-shot
transfer to DAPHNET, applying the identical percentile-rank-averaging *procedure* independently on
DAPHNET's pooled window scores (no DAPHNET-side tuning; ranks relative to DAPHNET's own 15,916-window
pool, not a shared frozen Voisard reference), this ordering reverses: reconstruction
error attains the higher point estimate (AUC 0.7046 vs. 0.5314 for the ensemble and 0.5884 for IF), though the
95% confidence intervals overlap. We deliberately did not retune fusion weights on DAPHNET, since
doing so would use target-domain performance to select the model — precisely the leakage a
zero-shot evaluation is designed to exclude. Fusion membership itself was source-locked from Voisard OOF diagnostics (not pre-registered before those diagnostics). The two ensembles therefore share membership and the rank-average formula, but not a strictly identical numeric scale. We report the reversal as-is.

One plausible explanation is that the latent-space one-class boundaries (IF, OCSVM) are fit to
the shape of Voisard's healthy latent distribution — a function of its sensor layout, population,
and acquisition protocol — and that boundary does not transfer cleanly to DAPHNET's zero-padded,
single-sensor input and different patient population. Reconstruction error, by contrast, is a
coarser, more distribution-agnostic signal ("does this resemble normal gait at all") and appears
to degrade more gracefully under this domain shift. We present this as a plausible mechanism,
not a demonstrated one; disentangling it from the identifiability limitation below would require
FOG-positive data from more than one subject.

## 6. Reproducibility and reporting contribution

A practical contribution is the explicit reproducibility pathway: stage-structured execution, YAML configuration, deterministic seeding (`PYTHONHASHSEED=42` before Python start), lockfile-pinned CI installs, and containerized reruns. For clinical-ML literature, these engineering details affect trust and independent verification.

## 7. Clinical interpretation boundary

These findings support pathology-tier gait screening as a research-oriented stratification aid, **not** direct prospective fall prediction. Labels are cohort-level diagnostic categories, not participant-level adjudicated future falls. Reported metrics are internal discriminative evidence within this dataset context, not proof of bedside clinical efficacy.

## 8. Limitations and future directions

Key limitations: retrospective single-dataset design, no prospective fall-outcome endpoint, no external multi-site validation, and single-trial API inference that does not fully reproduce multi-trial participant aggregation used during training. Future work: (i) prospective validation with incident falls, (ii) external cohort replication with frozen models, (iii) cohort-shift calibration, (iv) harmonized deep/tabular reporting on identical filtered endpoints.

## 9. Conclusion

The pipeline demonstrates participant-grouped, reproducible wearable-IMU modeling for multiclass pathology-tier screening on a heterogeneous open clinical dataset. Final conclusions on discrimination strength, interpretability, and sensor efficiency must be updated from regenerated artifacts before submission.
