# IMU Input Data Specification

## 1. Introduction

This document defines the data collection and input structure for the IMU-based fall-risk screening pipeline. The system analyzes raw wearable-sensor signals and performs feature engineering internally before model inference.

## 2. Sensor Configuration

The default setup uses four inertial measurement units (IMUs):

- **Head IMU**: upper-body orientation and abrupt movement response
- **Lower-back IMU**: core trunk motion and stability
- **Left-foot IMU**: gait-cycle and step timing behavior
- **Right-foot IMU**: gait symmetry and complementary step dynamics

This 4-sensor setup is the full reference configuration for feature extraction and evaluation.

## 3. Sensor Data Format

Each sensor is provided as one CSV file with the following canonical schema:

```csv
timestamp,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,mag_x,mag_y,mag_z
```

### 3.1 Column definitions

| Column | Description | Unit |
|---|---|---|
| `timestamp` | Sampling time | seconds |
| `acc_x`, `acc_y`, `acc_z` | Linear acceleration along X/Y/Z | m/s² |
| `gyr_x`, `gyr_y`, `gyr_z` | Angular velocity along X/Y/Z | rad/s |
| `mag_x`, `mag_y`, `mag_z` | Magnetic field along X/Y/Z | uT |

### 3.2 Data quality notes

- All sensors should follow the same schema.
- Signals should be time-synchronized across sensors per trial.
- Sampling frequency should remain consistent within each trial.

## 4. Metadata Specification

Each trial includes one JSON metadata file with contextual information:

```json
{
  "trial_id": "string",
  "participant_id": "string",
  "cohort": "string",
  "fall_probability": "number",
  "risk_label": "number",
  "laterality_biased": "boolean"
}
```

### 4.1 Metadata fields

| Field | Description |
|---|---|
| `trial_id` | Unique trial identifier |
| `participant_id` | Unique participant identifier |
| `cohort` | Cohort label (for example pathology group) |
| `fall_probability` | Continuous risk-related value from cohort mapping |
| `risk_label` | Binary or multiclass risk tier label |
| `laterality_biased` | Whether laterality bias is flagged for the cohort |

## 5. Data Structure Summary

| Component | Count |
|---|---:|
| IMU sensors | 4 |
| Columns per sensor CSV | 10 |
| Total sensor columns | 40 |
| Metadata fields | 6 |
| Total input fields | 46 |

Each trial is expected to provide:

- 4 sensor CSV files (one per IMU position)
- 1 metadata JSON file

## 6. Pipeline Context

The input schema feeds the following high-level stages:

1. Data ingestion (multi-sensor loading and validation)
2. Preprocessing (filtering, alignment, quality checks)
3. Feature extraction (temporal, spectral, trunk, orientation, asymmetry, turning)
4. Model inference (risk probabilities and class predictions)

## 7. Assumptions and Constraints

- Sensors are calibrated before recording.
- Timestamps are valid and non-corrupted.
- Units are preserved as specified.
- Trials are independent and correctly labeled.

## 8. Notes on Extensions

Potential extensions include:

- Real-time streaming sensor integration
- Additional physiological modalities
- Mobile/wearable deployment interfaces
- Participant-adaptive personalized modeling

---

Prepared by Kevin Mevada.
