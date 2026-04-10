# GaitGuard Example Data Files

This directory contains example files showing the required format for uploading IMU data to the GaitGuard fall risk analysis system.

## Required Files

### 1. Sensor Data Files (CSV format)
- **head_raw.csv** - Head sensor acceleration and gyroscope data
- **lower_back_raw.csv** - Lower back sensor acceleration and gyroscope data  
- **left_foot_raw.csv** - Left foot sensor acceleration and gyroscope data
- **right_foot_raw.csv** - Right foot sensor acceleration and gyroscope data

### 2. Metadata File (JSON format)
- **metadata.json** - Trial and participant information

## CSV File Format

Each sensor CSV must contain these columns:
- `PacketCounter` - Timestamp/sequence number (float)
- `Acc_X/Y/Z` - Acceleration data for X, Y, Z axes (float)
- `Gyr_X/Y/Z` - Gyroscope data for X, Y, Z axes (float)

Example sensor naming:
- Head: `HE_Acc_X`, `HE_Acc_Y`, `HE_Acc_Z`, `HE_Gyr_X`, `HE_Gyr_Y`, `HE_Gyr_Z`
- Lower Back: `LB_Acc_X`, `LB_Acc_Y`, `LB_Acc_Z`, `LB_Gyr_X`, `LB_Gyr_Y`, `LB_Gyr_Z`
- Left Foot: `LF_Acc_X`, `LF_Acc_Y`, `LF_Acc_Z`, `LF_Gyr_X`, `LF_Gyr_Y`, `LF_Gyr_Z`
- Right Foot: `RF_Acc_X`, `RF_Acc_Y`, `RF_Acc_Z`, `RF_Gyr_X`, `RF_Gyr_Y`, `RF_Gyr_Z`

## JSON Metadata Format

Required fields:
- `participant_id` - Unique participant identifier
- `trial_id` - Unique trial identifier  
- `cohort` - One of: "healthy", "elderly", "post_surgical", "neurological"
- `sampling_rate` - Sensor sampling frequency in Hz (typically 100)

Optional fields:
- `duration_seconds` - Trial duration
- `age` - Participant age
- `gender` - Participant gender (M/F)
- `height_cm` - Height in centimeters
- `weight_kg` - Weight in kilograms
- `notes` - Additional trial notes

## Upload Methods

1. **ZIP File**: Package all 5 files into a single ZIP archive
2. **Individual Files**: Upload the 4 CSV files + 1 JSON file separately

## Data Requirements

- **Sampling Rate**: 100 Hz recommended
- **Duration**: Minimum 10 seconds of continuous data
- **Units**: 
  - Acceleration: m/s²
  - Gyroscope: rad/s
  - Timestamp: seconds
- **Coordinate System**: Right-handed coordinate system aligned with body axes

## Processing Pipeline

The system will automatically:
1. Validate file formats and required columns
2. Extract 41 gait features from raw sensor data
3. Apply ensemble ML models for fall risk prediction
4. Generate comprehensive risk report with SHAP explanations
