# Pipeline reproducibility and RNG seeds

## Master seed

The canonical seed is **`reproducibility.seed`** in `configs/pipeline_config.yaml` (default **42**).

`python main.py` calls `set_global_seed()` immediately after loading config:

| Library | What is seeded |
|---------|----------------|
| `random` | Python stdlib RNG |
| `numpy` | Legacy `np.random` (used by many sklearn paths) |
| `torch` | CPU/CUDA weights and ops when PyTorch is installed |
| `PYTHONHASHSEED` | Overwritten to match `reproducibility.seed` (warns if the process started with a different value) |

Keep **`models.evaluation.random_state`** aligned with **`reproducibility.seed`** (both 42 by default). Training, feature selection, nested CV, Optuna TPE, bootstrap tests, and ensemble builders read `evaluation.random_state`. If the two values differ, `main.py` logs a warning.

### `PYTHONHASHSEED` (launch from the shell)

Python fixes hash-based dict/set ordering at **interpreter startup**. Docker and CI often pre-set `PYTHONHASHSEED=random`, which makes RFECV and other hash-iteration paths non-deterministic across environments even when NumPy/sklearn seeds match.

**Always launch the pipeline with an explicit hash seed:**

```bash
export PYTHONHASHSEED=42   # must match reproducibility.seed
cd fall_risk_pipeline
python main.py --stage all
```

On Windows PowerShell:

```powershell
$env:PYTHONHASHSEED = "42"
python main.py --stage all
```

`set_global_seed()` in `main.py` assigns `os.environ["PYTHONHASHSEED"] = str(seed)` (overriding `random` or other values) and emits a warning when it replaces a pre-existing value. That helps child processes but **does not retroactively fix** hash order in the already-running interpreter — export the variable before starting Python for bit-identical runs.

## Per-stage behavior

| Stage | Stochastic? | Seed source |
|-------|-------------|-------------|
| `ingest`, `validate_gait_events`, `preprocess` | No | Deterministic transforms (`signal_processor` uses filters/peaks only) |
| `eda` | Yes (row sample, t-SNE) | `get_pipeline_seed(config)` on `EDAAnalyzer`; t-SNE uses full-dataset `StandardScaler` (caption on figure — MED-004) |
| `features`, `select_features` | Mostly no | Feature selection CV uses `evaluation.random_state` |
| `train`, `evaluate`, `predict`, `report` | Yes | `evaluation.random_state`; Optuna `TPESampler(seed=…)` |
| `anomaly` | Yes (Isolation Forest) | `get_pipeline_seed(config)` |
| `deep_learning` (if enabled) | Yes | Per-LOSO-fold Optuna LR search when `loso_hyperparameter_tuning.enabled` (default true); global torch seed + `DataLoader(..., generator=…)` |

## Recommended full run

For the closest match across machines and reruns:

```bash
export PYTHONHASHSEED=42   # same value as reproducibility.seed — set before python starts
export OMP_NUM_THREADS=1
cd fall_risk_pipeline
python main.py --stage all
```

- **`PYTHONHASHSEED`**: must be exported **before** the Python process starts for stable hash-based iteration order. Do not rely on in-process overrides alone.
- **`OMP_NUM_THREADS=1`**: reduces numerical drift from parallel BLAS in some environments (optional but helpful).

With CUDA and `reproducibility.deterministic_torch: true`, cuDNN runs in deterministic mode (`benchmark=False`). Some GPU kernels may still differ slightly by driver/hardware.

## What is not guaranteed

- **Multi-process / `n_jobs > 1`**: joblib/loky worker order can add run-to-run variance even with a fixed seed.
- **Floating point**: different CPUs, BLAS builds, or library versions can change the last bits of metrics.
- **Optuna + parallel trials**: if trial execution order changes, hyperparameter search paths may diverge.
- **Third-party updates**: upgrading XGBoost, LightGBM, sklearn, or PyTorch can change defaults or algorithms.

For publication, record: git commit, `pipeline_config.yaml`, `PYTHONHASHSEED`, library versions (`pip freeze`), and whether the full pipeline or individual stages were run.

After `report`, compare `results/metrics/PIPELINE_VERSION.json` against your checkout:

- `git.commit` should match `git rev-parse HEAD` (and `dirty` should be `false` for frozen submissions).
- `config_sha256` / `config_file_sha256` detect config drift relative to the run that produced `metrics.csv`.
- `feature_selection.rfecv_importance_method` and `primary_endpoint` summarize settings that strongly affect reported AUCs.

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

## CI and Docker (`PYTHONHASHSEED`)

GitHub Actions (`.github/workflows/ci.yml`), the training `Dockerfile.pipeline`, and `Dockerfile.api` set **`PYTHONHASHSEED=42`**, matching `reproducibility.seed` in `pipeline_config.yaml`. Unit tests do not depend on hash iteration order, but aligning CI/Docker with the pipeline seed avoids environment drift.

For bit-identical full pipeline reruns, still export `PYTHONHASHSEED=42` **before** starting Python (see above).
