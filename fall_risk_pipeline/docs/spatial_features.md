# Spatial gait features — Phase 1 IMU integration

## What the pipeline computes

`phase1_spatiotemporal.py` (wired in `feature_extractor.py` when **left_foot** and **right_foot** are present) derives literature-aligned spatiotemporal and variability metrics from heel-strike / toe-off events and swing-phase foot acceleration.

| Feature | Method | Literature |
|---------|--------|------------|
| Stride duration (s) | Same-foot HS → HS interval | Trabassi, Moon, Voisard |
| Step duration (s) | Alternating L/R HS intervals | Voisard, Sadeghsalehi |
| Stance / swing (%) | HS→TO / TO→next HS as % of stride | Trabassi, Schlachetzki |
| Step / stride length (m) | Double integration of swing-phase acc + endpoint drift correction | Moon, Li, Voisard |
| Gait velocity (m/s) | step_length / swing duration | Yona, Trabassi |
| CV% | (SD/Mean)×100; rolling mean over stride windows | Moon, Trabassi |
| Symmetry index (SI%) | \|L−R\| / (0.5×(L+R)) × 100 | Voisard, Schlachetzki |

Legacy column names (`stride_time_mean`, `stance_phase_ratio`, `stride_time_cv` as a fraction) are still emitted for backward-compatible ablations.

## DAPHNET / LB-only trials

DAPHNET trials mapped to **lower_back only** do not have foot sensors. Phase 1 foot-based features are **not computed** for those trials (spatial/temporal foot columns remain absent or NaN at patient aggregation).

## Calibration caveats (Methods / Limitations)

Absolute step length and gait speed from IMU double integration are **uncalibrated** — no ground-truth stride length in the Figshare metadata. Drift correction (linear velocity detrend between swing endpoints) reduces but does not remove integration bias.

Recommended wording:

> Step length and gait velocity were estimated from swing-phase foot accelerometer double integration with endpoint drift correction (ZUPT at heel strike). These spatial proxies are reported for screening comparability with prior IMU gait studies (Moon 2020; Trabassi 2022) and should not be interpreted as laboratory-grade kinematics without external validation.

## Config

`pipeline_config.yaml`:

```yaml
features:
  phase1_spatiotemporal:
    enabled: true
    rolling_cv_window_strides: 5
    enable_spatial_integration: true
    integration_axis: acc_resultant   # or acc_x if AP-aligned
  spatial:
    - step_length_m
    - stride_length_m
    - gait_velocity_m_s
```

Set `enable_spatial_integration: false` to emit timing/variability/SI only (no length or velocity).

## Optional future work

- Anthropometric priors (height, leg length) for scale calibration  
- Magnetometer- or UWB-assisted drift correction  
- Pace-conditioned velocity (slow / self-selected / fast) when trial metadata includes speed condition
