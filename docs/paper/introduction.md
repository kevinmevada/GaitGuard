# Introduction

Falls are a major source of morbidity, loss of independence, and healthcare burden in older adults and in people with neurological or orthopedic gait impairment. Clinical fall-risk assessment remains essential, but routine workflows are often constrained by brief bedside screens, inter-rater variability, and limited access to longitudinal movement data outside specialized laboratories. Wearable inertial measurement units (IMUs) offer a practical route to objective gait quantification because they are portable, relatively low cost, and compatible with repeated measurements across clinical and community settings.

Recent IMU-based studies report promising fall-related discrimination performance; however, three methodological limitations repeatedly reduce translational confidence. First, many pipelines evaluate at trial level with random splits that can leak participant-specific patterns between train and test sets, inflating apparent performance. Second, several approaches collapse heterogeneous pathologies into broad binary labels, obscuring clinically meaningful differences between orthopedic and neurological gait signatures. Third, reproducibility is frequently limited by incomplete code release, unclear preprocessing assumptions, or missing execution scaffolding that prevents one-command re-runs by independent groups.

The open multi-cohort dataset released by Voisard et al. (Figshare DOI: 10.6084/m9.figshare.28806086) creates an opportunity to address these gaps with a unified, transparent benchmark. It includes 1,355 walking trials from 260 participants across eight cohorts and four synchronized IMU locations (head, lower back, left foot, right foot). This composition supports direct comparison across healthy, orthopedic, and neurological groups while preserving participant-level structure for leakage-safe evaluation. A concurrent independent study (Sadeghsalehi et al., 2025) also uses this dataset and identifies the same cohort laterality confound; that work evaluates four binary screening tasks, whereas this work evaluates the full eight-cohort multi-class tier under strict participant-grouped leave-one-subject-out validation, adds a healthy-reference anomaly screening ensemble, and adds zero-shot cross-dataset transfer to an external freezing-of-gait dataset.

In this work, we present a reproducible pathology-tier gait screening pipeline that combines classical machine learning, deep learning baselines, and Healthy-reference anomaly analysis on this dataset. The **primary** endpoint is unsupervised BiLSTM-AE anomaly screening (Isolation Forest + one-class SVM on learned latents, source-locked rank-averaged ensemble) under LOSO. A **secondary** supervised endpoint is three-class pathology-tier discrimination (healthy, orthopedic, neurological), used as a supplementary stratification benchmark — not as a substitute for the primary anomaly endpoint. We use participant-grouped validation, including leave-one-subject-out (LOSO) evaluation, to estimate generalization under strict subject independence.

Our pipeline performs end-to-end ingestion, preprocessing, feature extraction, dimensionality control, model training, evaluation, and report generation. Feature engineering spans temporal, spectral, trunk dynamics, orientation, and asymmetry domains, followed by grouped RFECV/SHAP-guided selection with an explicit cap on final predictor count. The tabular branch trains and compares XGBoost, LightGBM, Random Forest, SVM, and MLP models with soft-voting ensemble integration; the deep branch benchmarks InceptionTime, Transformer, TCN, CNN-1D, and BiLSTM-attention architectures under the same cohort structure.

This study makes four primary contributions for Sensors-style reproducible research:

1. A healthy-reference anomaly screening ensemble (BiLSTM-autoencoder latent representations scored by Isolation Forest and One-Class SVM) requiring no pathological training data, evaluated under strict LOSO across all eight Voisard cohorts.
2. A source-locked, zero-shot cross-dataset transfer evaluation to an external freezing-of-gait dataset (DAPHNET), using fusion membership fixed entirely from source-domain evidence, with no target-domain tuning.
3. Full methodological transparency regarding known confounds and limitations discovered during development, including a cohort laterality confound (also independently identified by Sadeghsalehi et al., 2025) and a data-contamination issue in an early version of the supervised tabular pipeline, both disclosed with quantified impact.
4. A fully documented, executable end-to-end pipeline with participant-grouped evaluation, interpretability analyses (SHAP, feature/sensor ablation; tabular ablations provisional), and publicly inspectable artifacts intended to support independent replication.

We emphasize that labels in this dataset are cohort-level pathology categories rather than prospectively adjudicated individual fall outcomes. Accordingly, performance metrics should be interpreted as evidence for pathology-tier screening utility, not direct clinical fall prediction. Prospective external validation with participant-level incident falls is required before clinical deployment claims.

## Suggested citation anchors for this section

- Fall epidemiology and burden in older adults.
- Wearable IMU gait assessment and mobility impairment.
- Methodological risk of subject leakage in human-sensor ML.
- Voisard et al. dataset paper and Figshare dataset DOI.
- Prior IMU fall-risk studies used for comparative context in Discussion/Table 2.
- Sadeghsalehi et al. (2025) concurrent Voisard-dataset screening study.
