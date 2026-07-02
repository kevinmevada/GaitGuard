"""Local filesystem staging helpers (replaces OSDF/stashcp when GAITGUARD_LOCAL_STAGING is set)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def local_staging_enabled() -> bool:
    return bool(os.environ.get("GAITGUARD_LOCAL_STAGING"))


def gaitguard_root() -> Path:
    """Canonical local gaitguard tree (raw, processed, features, hpc/)."""
    root = os.environ.get("GAITGUARD_LOCAL_STAGING")
    if not root:
        raw = os.environ.get("OSDF_GAITGUARD", "/ospool/ap40/data/kevin.mevada/gaitguard")
        if raw.startswith("osdf://"):
            raw = raw[len("osdf://") :]
        return Path(raw)
    return Path(root)


def local_copy(src: Path, dst: Path, *, recursive: bool = False) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    if src.is_dir() and recursive:
        shutil.copytree(src, dst)
        return True
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    return False


def osd_uri_to_local_path(uri: str) -> Path | None:
    """Map osdf://.../gaitguard/... or /ospool/.../gaitguard/... to local gaitguard root."""
    if not local_staging_enabled():
        return None
    gg = gaitguard_root()
    marker = "/gaitguard/"
    if marker in uri:
        rel = uri.split(marker, 1)[1].split("?")[0].strip("/")
        return gg / rel
    if uri.startswith(str(gg)):
        return Path(uri.split("?")[0])
    return None
