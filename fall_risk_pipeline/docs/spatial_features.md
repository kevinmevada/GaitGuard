# Spatial gait features — not extracted

## What the pipeline computes instead

`feature_extractor.py` derives **temporal** and **bilateral asymmetry** quantities from foot IMU heel-strike / toe-off events and trunk/head signals. It does **not** estimate absolute step length, gait speed, or step width.

| Available (trial level) | Source |
|-------------------------|--------|
| Stride time mean / std / CV | Foot `heel_strike_*` intervals |
| Cadence, step count | Heel-strike rate |
| Stance phase ratio | Heel strike → toe off |
| Stride-time asymmetry (mean / std) | Left vs right stride statistics |
| RMS acceleration asymmetry | Left vs right foot resultant |

## Why step length and gait speed are omitted

Absolute **step length** and **gait speed** need either:

- Known anthropometric calibration or external distance reference, or  
- Double integration of acceleration / magnetometer-based displacement (high drift without additional sensors or constraints).

The Figshare cohort provides 100 Hz IMU on head, lower back, and feet — **no ground-truth step length** in trial metadata. Reporting uncalibrated spatial metrics would be misleading for a clinical paper.

## Config and README

`pipeline_config.yaml` keeps `spatial: []` with a comment pointing here. Do not cite step length or gait speed in methods unless you add a validated estimator and re-run `features`.

## Suggested paper wording (Methods / Limitations)

> Spatial parameters (step length, gait speed, step width) were not derived from IMU alone. Trial features consisted of temporal gait-cycle metrics (stride time, cadence, stance ratio), trunk and spectral dynamics, orientation-based postural measures, and bilateral asymmetry indices computed from foot-mounted accelerometers.

## Optional future work

If height or leg length is added to `trial_metadata.csv`, a coarse speed proxy could be implemented as  
`gait_speed ≈ step_length_prior × cadence` with explicit uncertainty — not enabled in the current pipeline.
