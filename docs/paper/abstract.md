# Paper draft — title and abstract

**Journal target:** *Sensors* (MDPI)  
**Status:** Abstract reconciled for Overleaf manuscript (`docs/paper/overleaf/`).  
**Keywords:** gait analysis; wearable sensors; inertial measurement unit; anomaly detection; leave-one-subject-out; cross-dataset transfer; freezing of gait; clinical machine learning

---

## Working title

**Healthy-Reference Anomaly Screening and Cross-Dataset Transfer for Wearable Gait Monitoring: A Leave-One-Subject-Out Study Across Eight Clinical Cohorts**

*Alternates:*  
- *Healthy-Reference Gait Anomaly Screening with Wearable IMUs: Leave-One-Subject-Out Evaluation Across Eight Clinical Cohorts (N = 260)*  
- *GaitGuard: Ensemble One-Class Anomaly Screening with Supplementary Pathology-Tier Supervised Models*

---

## Abstract (*Sensors*)

Wearable inertial measurement units (IMUs) enable continuous, low-cost gait monitoring outside clinical settings, but most existing anomaly-detection approaches are trained and validated on single, homogeneous patient populations, limiting evidence of generalization. We present a healthy-reference gait anomaly screening system trained exclusively on gait data from healthy participants and evaluated under strict participant-grouped leave-one-subject-out (LOSO) cross-validation across a public, de-identified, eight-cohort clinical gait dataset (260 participants, 1,355 trials; Voisard et al., 2025), spanning healthy, orthopedic (hip osteoarthritis, knee osteoarthritis, ACL injury), and neurological (Parkinson's disease, cerebrovascular accident, chemotherapy-induced peripheral neuropathy, radiculopathy) cohorts. A two-method ensemble — Isolation Forest and One-Class SVM applied to BiLSTM-autoencoder latent representations — achieves an out-of-fold ROC-AUC of 0.7545 (Isolation Forest alone 0.7540; empirically IF-dominated), outperforming a rejected three-method variant that included raw reconstruction error (AUC 0.6238). As a secondary, zero-shot cross-dataset check — no target-domain fine-tuning — the same source-locked fusion *membership* was applied to an external freezing-of-gait dataset (DAPHNET; single lower-back sensor; 9 participants) with identical per-domain percentile rank-averaging; under this domain shift, AE reconstruction error outperformed the latent one-class ensemble (AUC 0.7046 vs. 0.5314), reversing the in-domain ordering, though all 62 freezing-positive windows originated from a single participant, limiting this transfer result to a preliminary, single-subject case study rather than evidence of cross-subject generalization. We additionally report supervised pathology-tier classification results (macro one-vs-rest AUC 0.84) as provisional, following the discovery and correction of a data-contamination issue affecting the tabular pipeline (see Methods and Limitations). We disclose all identified methodological limitations, including a laterality confound shared with a concurrent independent study on the same dataset, to support transparent, reproducible evaluation of wearable gait screening systems.

---

## Metrics fill-in

_Auto-updated from pipeline artifacts — do not edit manually._

| Metric | Value |
|--------|-------|
| Primary BiLSTM-AE MCC (LOSO OOF) | 0.388 |
| Primary BiLSTM-AE AUROC (LOSO OOF) | 0.7545 |
| Primary BiLSTM-AE F1 | 0.786 |
| Primary BiLSTM-AE Sensitivity (Youden) | 0.711 |
| Primary BiLSTM-AE Specificity (Youden) | 0.720 |
| Primary BiLSTM-AE Cohen κ | 0.369 |
| Primary BiLSTM-AE PR-AUC | 0.8669 |
| Secondary deployable ensemble macro OvR AUC | 0.8404 (ensemble_soft_voting; provisional) |
| Best supervised single-model LOSO macro OvR AUC | 0.8415 (xgboost; provisional) |
| MCC abstract-lead threshold | 0.70 |

_Headline: report **AUROC** as threshold-independent headline; cite **MCC** and Cohen κ for rigor._

Regenerate after each pipeline run:

```bash
cd fall_risk_pipeline && python main.py
python ../scripts/regenerate_paper_results.py
```
