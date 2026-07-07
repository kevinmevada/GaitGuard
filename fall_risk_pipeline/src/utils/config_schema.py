"""
Configuration schema validation (additive, non-breaking).

This module validates the loaded ``pipeline_config.yaml`` dict against a
declarative schema *before* any pipeline stage runs, so a missing or
malformed key fails in milliseconds with a clear, aggregated report —
instead of failing hours into a run with a bare ``KeyError``/``TypeError``
at whatever point in the pipeline first happens to touch that key.

Design goals (all deliberate):

1. **Additive, not a rewrite.** Every existing ``config["a"]["b"]["c"]``
   access pattern throughout the codebase keeps working unchanged; this
   module only *validates* the dict shape, it does not replace the dict
   with a typed object.
2. **Fail together, not one at a time.** ``validate_config`` collects
   every problem it finds and reports them all at once, so a user does
   not have to fix one missing key, rerun, discover the next missing key,
   and repeat.
3. **Forward-compatible.** Only the keys declared in the schema below are
   checked. Unknown extra keys are always allowed (they are not an error)
   so this does not become a maintenance burden that has to track every
   experimental config addition.
4. **Opt-out escape hatch.** ``main.py`` calls this by default; a
   ``GAITGUARD_SKIP_CONFIG_VALIDATION=1`` environment variable (or
   ``--skip-config-validation`` CLI flag) bypasses it for advanced users
   iterating on a config shape not yet reflected in the schema below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


class ConfigValidationError(ValueError):
    """Raised when ``pipeline_config.yaml`` fails schema validation.

    Carries every problem found (not just the first) in ``.errors``.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        joined = "\n  - ".join(errors)
        super().__init__(
            f"pipeline_config.yaml failed validation ({len(errors)} problem"
            f"{'s' if len(errors) != 1 else ''}):\n  - {joined}\n\n"
            "Fix the config above, or set GAITGUARD_SKIP_CONFIG_VALIDATION=1 "
            "to bypass this check (not recommended for a full pipeline run)."
        )


@dataclass
class Field:
    """One declared field in the schema.

    ``path`` is a dotted path like ``"paths.processed_data"``.
    ``type_`` may be a type or tuple of types (as for ``isinstance``).
    ``required`` — if True and the key is absent, that's an error.
    ``validator`` — optional extra check ``(value) -> str | None``;
        return an error message string, or None if the value is fine.
    """

    path: str
    type_: type | tuple[type, ...]
    required: bool = True
    validator: Callable[[Any], str | None] | None = None


def _non_empty_list(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) == 0:
        return "must be a non-empty list"
    return None


def _in_range_0_1(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
        return "must be a number between 0 and 1"
    return None


def _positive_number(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or float(value) <= 0:
        return "must be a positive number"
    return None


def _known_label_mode(value: Any) -> str | None:
    if value not in ("binary", "multiclass"):
        return f"must be 'binary' or 'multiclass', got {value!r}"
    return None


# ---------------------------------------------------------------------------
# The schema itself. Deliberately covers the sections most load-bearing for
# pipeline correctness (paths, reproducibility, dataset, feature_selection,
# models, deep_learning, preprocessing) rather than every leaf key in the
# 630+ line config — see module docstring point 3 (forward-compatible).
# ---------------------------------------------------------------------------
SCHEMA: list[Field] = [
    # Reproducibility
    Field("reproducibility.seed", int),
    # Paths — every value must be a non-empty string (a filesystem path)
    Field("paths.raw_data", str),
    Field("paths.processed_data", str),
    Field("paths.features", str),
    Field("paths.results", str),
    Field("paths.metrics", str),
    Field("paths.checkpoints", str),
    Field("paths.logs", str),
    # Dataset
    Field("dataset.sensor_positions", list, validator=_non_empty_list),
    Field("dataset.sampling_rate", (int, float), validator=_positive_number),
    Field("dataset.label_mode", str, validator=_known_label_mode),
    Field("dataset.cohort_labels", dict),
    Field(
        "dataset.subject_split.healthy_train_fraction",
        (int, float),
        validator=_in_range_0_1,
    ),
    Field(
        "dataset.subject_split.healthy_val_fraction",
        (int, float),
        validator=_in_range_0_1,
    ),
    Field(
        "dataset.subject_split.healthy_test_fraction",
        (int, float),
        validator=_in_range_0_1,
    ),
    # Preprocessing
    Field("preprocessing.unified_acc_bandpass.low_hz", (int, float), validator=_positive_number),
    Field("preprocessing.unified_acc_bandpass.high_hz", (int, float), validator=_positive_number),
    Field("preprocessing.min_trial_length_s", (int, float), validator=_positive_number),
    # Feature selection
    Field("feature_selection.max_features", int, validator=_positive_number),
    Field("feature_selection.min_features", int, validator=_positive_number),
    # Models (tabular)
    Field("models.run", list, validator=_non_empty_list),
    Field("models.evaluation.strategy", str),
    Field("models.evaluation.random_state", int),
    Field(
        "models.evaluation.cohort_auc_min_n",
        int,
        required=False,
        validator=_positive_number,
    ),
    # Deep learning
    Field("deep_learning.enabled", bool),
    Field("deep_learning.sequence_length", int, required=False, validator=_positive_number),
    Field("deep_learning.overlap", (int, float), required=False, validator=_in_range_0_1),
]


def _get_nested(config: dict[str, Any], dotted_path: str) -> tuple[bool, Any]:
    """Return (found, value) for a dotted path into a nested dict."""
    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate ``config`` against :data:`SCHEMA`.

    Returns a list of human-readable error strings (empty if valid).
    Never raises — use :func:`validate_config_or_raise` for the
    fail-fast entry point.
    """
    errors: list[str] = []
    for f in SCHEMA:
        found, value = _get_nested(config, f.path)
        if not found:
            if f.required:
                errors.append(f"'{f.path}' is required but missing")
            continue
        if not isinstance(value, f.type_):
            expected = (
                f.type_.__name__
                if isinstance(f.type_, type)
                else " or ".join(t.__name__ for t in f.type_)
            )
            errors.append(
                f"'{f.path}' must be of type {expected}, got "
                f"{type(value).__name__} ({value!r})"
            )
            continue
        if f.validator is not None:
            msg = f.validator(value)
            if msg is not None:
                errors.append(f"'{f.path}' {msg} (got {value!r})")

    # A few cross-field checks that don't fit the single-field Field model.
    found_fracs, _ = _get_nested(config, "dataset.subject_split")
    if found_fracs:
        fracs = []
        for key in ("healthy_train_fraction", "healthy_val_fraction", "healthy_test_fraction"):
            found, v = _get_nested(config, f"dataset.subject_split.{key}")
            if found and isinstance(v, (int, float)):
                fracs.append(float(v))
        if len(fracs) == 3 and abs(sum(fracs) - 1.0) > 1e-6:
            errors.append(
                "'dataset.subject_split' train/val/test fractions must sum to "
                f"1.0, got {sum(fracs):.4f} ({fracs})"
            )

    return errors


def validate_config_or_raise(config: dict[str, Any]) -> None:
    """Fail-fast entry point: raises :class:`ConfigValidationError` if invalid.

    Honors ``GAITGUARD_SKIP_CONFIG_VALIDATION=1`` as an escape hatch.
    """
    if os.environ.get("GAITGUARD_SKIP_CONFIG_VALIDATION", "").strip() in ("1", "true", "True"):
        return
    errors = validate_config(config)
    if errors:
        raise ConfigValidationError(errors)
