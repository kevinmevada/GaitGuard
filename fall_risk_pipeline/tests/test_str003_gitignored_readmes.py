"""STR-003: tracked placeholder READMEs for gitignored output directories."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"


def _negated_readme_paths() -> list[Path]:
    paths: list[Path] = []
    for line in GITIGNORE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("!") and stripped.endswith("README.md"):
            rel = stripped[1:].lstrip("/")
            paths.append(REPO_ROOT / rel)
    return paths


def test_gitignored_placeholder_readmes_exist():
    missing = [p for p in _negated_readme_paths() if not p.is_file()]
    assert not missing, f"Missing placeholder READMEs: {missing}"


def test_placeholder_readmes_mention_generated_or_gitignore():
    for path in _negated_readme_paths():
        text = path.read_text(encoding="utf-8").lower()
        assert "gitignore" in text or "generated" in text, path.as_posix()


def test_gitignore_declares_results_readme_exceptions():
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "!fall_risk_pipeline/results/README.md" in text
    assert re.search(r"!fall_risk_pipeline/results/checkpoints/README\.md", text)
