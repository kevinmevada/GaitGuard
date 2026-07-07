"""Tests for the additive config-schema validation layer (src/utils/config_schema.py).

Covers: the real committed pipeline_config.yaml validates cleanly, common
misconfigurations are caught and aggregated (not one-at-a-time), and the
escape hatch environment variable works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.config_schema import (  # noqa: E402
    ConfigValidationError,
    validate_config,
    validate_config_or_raise,
)


@pytest.fixture
def real_config() -> dict:
    path = PIPELINE_ROOT / "configs" / "pipeline_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_real_committed_config_validates_cleanly(real_config):
    errors = validate_config(real_config)
    assert errors == [], f"Real config should validate with no errors, got: {errors}"


def test_missing_required_path_is_caught(real_config):
    broken = dict(real_config)
    broken["paths"] = {k: v for k, v in real_config["paths"].items() if k != "metrics"}
    errors = validate_config(broken)
    assert any("paths.metrics" in e for e in errors)


def test_multiple_problems_are_aggregated_not_one_at_a_time(real_config):
    broken = dict(real_config)
    broken["dataset"] = dict(real_config["dataset"])
    broken["dataset"]["sampling_rate"] = -100
    broken["models"] = dict(real_config["models"])
    broken["models"]["run"] = []
    del broken["paths"]

    errors = validate_config(broken)
    # All three distinct problems should be reported in a single pass.
    assert any("paths" in e for e in errors)
    assert any("sampling_rate" in e for e in errors)
    assert any("models.run" in e for e in errors)
    assert len(errors) >= 3


def test_wrong_type_is_caught(real_config):
    broken = dict(real_config)
    broken["dataset"] = dict(real_config["dataset"])
    broken["dataset"]["sensor_positions"] = "head"  # should be a list, not a string
    errors = validate_config(broken)
    assert any("sensor_positions" in e for e in errors)


def test_subject_split_fractions_must_sum_to_one(real_config):
    broken = dict(real_config)
    broken["dataset"] = dict(real_config["dataset"])
    broken["dataset"]["subject_split"] = dict(real_config["dataset"]["subject_split"])
    broken["dataset"]["subject_split"]["healthy_train_fraction"] = 0.9  # now sums to > 1
    errors = validate_config(broken)
    assert any("sum to" in e for e in errors)


def test_unknown_extra_keys_are_allowed(real_config):
    """Forward-compatibility: keys not in the schema must never be flagged."""
    extended = dict(real_config)
    extended["some_brand_new_experimental_section"] = {"anything": "goes"}
    errors = validate_config(extended)
    assert errors == []


def test_validate_config_or_raise_raises_with_aggregated_message(real_config):
    broken = dict(real_config)
    del broken["paths"]
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config_or_raise(broken)
    assert len(excinfo.value.errors) >= 1
    assert "paths" in str(excinfo.value)


def test_escape_hatch_env_var_bypasses_validation(real_config, monkeypatch):
    monkeypatch.setenv("GAITGUARD_SKIP_CONFIG_VALIDATION", "1")
    broken = dict(real_config)
    del broken["paths"]
    # Should not raise despite being invalid, because the escape hatch is set.
    validate_config_or_raise(broken)
