#!/bin/bash
# Print Condor submit commands for the full pipeline in dependency order.
# Run each line manually after the previous stage succeeds (check condor/logs).
set -euo pipefail

cd "$(dirname "$0")/.."

cat <<'EOF'
# Fresh rerun from raw data (after setup_ospool_staging.sh):
condor_submit condor/cpu_stage.sub -a "stage=discover"
condor_submit condor/cpu_stage.sub -a "stage=validate_raw"
condor_submit condor/ingest.sub

# After ingest completes:
condor_submit condor/cpu_stage.sub -a "stage=preprocess"
condor_submit condor/cpu_stage.sub -a "stage=validate_gait_events"
condor_submit condor/cpu_stage.sub -a "stage=eda"
condor_submit condor/cpu_stage.sub -a "stage=features"
condor_submit condor/cpu_stage.sub -a "stage=phase3_features"
condor_submit condor/cpu_stage.sub -a "stage=select_features"
condor_submit condor/cpu_stage.sub -a "stage=train"
condor_submit condor/cpu_stage.sub -a "stage=evaluate"
condor_submit condor/gpu_stage.sub -a "stage=train_deep"
condor_submit condor/cpu_stage.sub -a "stage=ablation"
condor_submit condor/gpu_stage.sub -a "stage=sensor_ablation"
condor_submit condor/cpu_stage.sub -a "stage=classical_baselines"
condor_submit condor/gpu_stage.sub -a "stage=anomaly"
condor_submit condor/gpu_stage.sub -a "stage=dl_baselines"
condor_submit condor/cpu_stage.sub -a "stage=competitor_metrics"
condor_submit condor/gpu_stage.sub -a "stage=severity_regression"
condor_submit condor/cpu_stage.sub -a "stage=statistical_benchmark"
condor_submit condor/cpu_stage.sub -a "stage=compute_overhead"
condor_submit condor/cpu_stage.sub -a "stage=novelty_table"
condor_submit condor/cpu_stage.sub -a "stage=per_cohort_loso"
condor_submit condor/cpu_stage.sub -a "stage=fall_risk_spearman"
condor_submit condor/cpu_stage.sub -a "stage=cross_cohort"
condor_submit condor/cpu_stage.sub -a "stage=predict"
condor_submit condor/cpu_stage.sub -a "stage=report"
EOF
