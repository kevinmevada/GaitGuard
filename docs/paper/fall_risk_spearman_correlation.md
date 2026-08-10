# Fall-risk clinical validation — Spearman correlation

Anomaly score is evaluated as a **proxy for fall risk** by correlating literature / metadata fall probability with BiLSTM-AE LOSO anomaly scores.

**Score column:** `bilstm_ae_ensemble_score`  
**Fall probability:** cohort reference rates (Voisard / clinical literature) or trial `fall_probability` from ingest metadata when present.

## Primary result — all participants

- **Spearman ρ** = 0.3332 (p = 0.0000, n = 260 participants, significant at α = 0.05)
- Mean anomaly score = 0.3959; mean reference fall probability = 32.7%

## Per pathological cohort (Healthy vs cohort contrast)

Within a single cohort, fall probability is cohort-constant, so ρ is reported for **Healthy + pathological tier** participants where fall probability varies.

| Cohort | vs Healthy | n participants | ρ | p-value | Mean score | Ref. fall prob. (%) |
|---|---|---:|---:|---:|---:|---:|
| **PD** | Healthy + PD | 97 | 0.2048 | 0.0442 | 0.3293 | 20.6 |
| **CVA** | Healthy + CVA | 122 | 0.4855 | 0.0000 | 0.4115 | 24.9 |
| **CIPN** | Healthy + CIPN | 92 | 0.1754 | 0.0944 | 0.3317 | 12.8 |
| **RIL** | Healthy + RIL | 124 | 0.3647 | 0.0000 | 0.3735 | 19.1 |
| **HOA** | Healthy + HOA | 88 | -0.0553 | 0.6087 | 0.3127 | 9.2 |
| **TKA** | Healthy + TKA | 91 | 0.1197 | 0.2583 | 0.3246 | 8.9 |
| **ACL** | Healthy + ACL | 84 | -0.2176 | 0.0468 | 0.3026 | 7.0 |

## Interpretation

- **Positive ρ** — higher literature fall-risk cohorts show higher anomaly scores (supports deployment as clinical decision-support signal).
- **AUROC/F1** answer discrimination; this table answers **clinical relevance**.
- Rows `within_{cohort}` in the CSV are NA by design (no fall-probability variance).

### Cohort reference fall probabilities (%)

| Cohort | Reference fall prob. (%) |
|---|---:|
| Healthy | 5.2 |
| PD | 67.3 |
| CVA | 54.2 |
| CIPN | 41.8 |
| RIL | 38.9 |
| HOA | 28.5 |
| TKA | 24.1 |
| ACL | 18.7 |
