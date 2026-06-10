# Gait event detection validation

Algorithm: peak detection on `-acc_z` (75th-percentile height, 0.3 s minimum spacing).
Ground truth: Figshare `gait_events.csv` or `leftGaitEvents` / `rightGaitEvents` in trial metadata.
Match tolerance: ±50 ms (±5 samples at 100 Hz).

Trials evaluated: 1356

## Summary (heel strike)

| Side | Trials | Precision | Recall | F1 | MAE (ms) |
|---|---:|---:|---:|---:|---:|
| left | 1356 | 0.130 | 0.260 | 0.173 | 29.1 |
| right | 1356 | 0.280 | 0.562 | 0.374 | 25.7 |
| both | 1356 | 0.205 | 0.411 | 0.273 | 27.2 |

Per-trial metrics: `gait_event_validation_by_trial.csv`
Aggregate metrics: `gait_event_validation_summary.csv`
