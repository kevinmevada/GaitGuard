"""SEC-014: Three.js addons loaded via import map with SRI integrity pins."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "api" / "static"


def test_main_js_uses_importmap_specifiers_not_raw_cdn():
    js = (FRONTEND / "main.js").read_text(encoding="utf-8")
    assert "await import('three')" in js
    assert "three/addons/loaders/GLTFLoader.js" in js
    assert "cdn.jsdelivr.net/npm/three" not in js


def test_importmap_integrity_covers_three_addons():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.S)
    assert match, "import map missing"
    data = json.loads(match.group(1))
    integrity = data.get("integrity", {})
    assert integrity
    required_suffixes = (
        "build/three.module.js",
        "loaders/GLTFLoader.js",
        "postprocessing/EffectComposer.js",
        "postprocessing/UnrealBloomPass.js",
        "shaders/CopyShader.js",
    )
    for suffix in required_suffixes:
        assert any(url.endswith(suffix) and val.startswith("sha384-") for url, val in integrity.items()), suffix
    assert len(integrity) >= 10
