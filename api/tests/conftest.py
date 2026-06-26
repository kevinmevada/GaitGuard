"""Ensure API tests import `api/main.py`, not `fall_risk_pipeline/main.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
API_MAIN_PATH = API_ROOT / "main.py"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _load_api_main():
    spec = importlib.util.spec_from_file_location("gaitguard_api_main", API_MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API module from {API_MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# When both pipeline and API tests are collected, `fall_risk_pipeline/main.py`
# shadows `api/main.py` on sys.path. Bind the API module for `from main import …`.
sys.modules["main"] = _load_api_main()
