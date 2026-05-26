# Pipeline reproducibility and RNG seeds

## Master seed

The canonical seed is **`reproducibility.seed`** in `configs/pipeline_config.yaml` (default **42**).

`python main.py` calls `set_global_seed()` immediately after loading config:

| Library | What is seeded |
|---------|----------------|
| `random` | Python stdlib RNG |
| `numpy` | Legacy `np.random` (used by many sklearn paths) |
| `torch` | CPU/CUDA weights and ops when PyTorch is installed |
| `PYTHONHASHSEED` | Set via `os.environ.setdefault` (best effort in-process) |

Keep **`models.evaluation.random_state`** aligned with **`reproducibility.seed`** (both 42 by default). Training, feature selection, nested CV, Optuna TPE, bootstrap tests, and ensemble builders read `evaluation.random_state`. If the two values differ, `main.py` logs a warning.

## Per-stage behavior

| Stage | Stochastic? | Seed source |
|-------|-------------|-------------|
| `ingest`, `validate_gait_events`, `preprocess` | No | Deterministic transforms (`signal_processor` uses filters/peaks only) |
| `eda` | Yes (row sample, t-SNE) | `get_pipeline_seed(config)` on `EDAAnalyzer` |
| `features`, `select_features` | Mostly no | Feature selection CV uses `evaluation.random_state` |
| `train`, `evaluate`, `predict`, `report` | Yes | `evaluation.random_state`; Optuna `TPESampler(seed=…)` |
| `anomaly` | Yes (Isolation Forest) | `get_pipeline_seed(config)` |
| `deep_learning` (if enabled) | Yes | Global torch seed + `DataLoader(..., generator=…)` in `DeepModelTrainer` |

## Recommended full run

For the closest match across machines and reruns:

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
cd fall_risk_pipeline
python main.py --stage all
```

- **`PYTHONHASHSEED`**: must be set **before** the Python process starts for stable hash-based iteration order.
- **`OMP_NUM_THREADS=1`**: reduces numerical drift from parallel BLAS in some environments (optional but helpful).

With CUDA and `reproducibility.deterministic_torch: true`, cuDNN runs in deterministic mode (`benchmark=False`). Some GPU kernels may still differ slightly by driver/hardware.

## What is not guaranteed

- **Multi-process / `n_jobs > 1`**: joblib/loky worker order can add run-to-run variance even with a fixed seed.
- **Floating point**: different CPUs, BLAS builds, or library versions can change the last bits of metrics.
- **Optuna + parallel trials**: if trial execution order changes, hyperparameter search paths may diverge.
- **Third-party updates**: upgrading XGBoost, LightGBM, sklearn, or PyTorch can change defaults or algorithms.

For publication, record: git commit, `pipeline_config.yaml`, `PYTHONHASHSEED`, library versions (`pip freeze`), and whether the full pipeline or individual stages were run.

## Changing the seed

Edit both in `configs/pipeline_config.yaml`:

```yaml
reproducibility:
  seed: 123

models:
  evaluation:
    random_state: 123
```

Then rerun affected stages (`eda` onward for plots; `train` onward for models and metrics).

## API / inference

The FastAPI service (`api/main.py`) does not retrain models; it loads fixed checkpoints. Reproducibility of **training** is governed by this pipeline entrypoint only.
