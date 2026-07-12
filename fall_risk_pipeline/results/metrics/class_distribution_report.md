# Class distribution report

**Level:** participant (N=269)
**Label mode:** `multiclass` — Primary target: 3-class labels (0=Healthy, 1=orthopedic, 2=neurological). Preserves separation between OA/ACL and neurodegenerative/vascular cohorts.

## Cohort → label mapping

| Cohort | Multiclass | Train label | Binary (≥1) | Binary (≥2) | Ref. fall % |
|---|---:|---:|---:|---:|---:|
| Healthy | 0 | 0 | 0 | 0 | 5.2 |
| RIL | 2 | 2 | 1 | 1 | 38.9 |
| CVA | 2 | 2 | 1 | 1 | 54.2 |
| PD | 2 | 2 | 1 | 1 | 67.3 |
| CIPN | 2 | 2 | 1 | 1 | 41.8 |
| KneeOA | 1 | 1 | 1 | 0 | 24.1 |
| HipOA | 1 | 1 | 1 | 0 | 28.5 |
| ACL | 1 | 1 | 1 | 0 | 18.7 |

## Training label counts (`label_mode=multiclass`)

- **low (Healthy)** (label=0): **73** (27.1%)
- **moderate (orthopedic: HipOA/KneeOA/ACL)** (label=1): **44** (16.4%)
- **high (neurological: PD/CVA/CIPN/RIL)** (label=2): **152** (56.5%)

Alternative binary collapses: see `binary_label_sensitivity.csv` and `docs/label_binning.md`.