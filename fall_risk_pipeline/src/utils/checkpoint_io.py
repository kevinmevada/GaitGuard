"""
Signed checkpoint serialization for sklearn pipelines.

Checkpoints are stored with joblib and registered in ``checkpoint_manifest.json``
(SHA-256 digest per file). When ``CHECKPOINT_HMAC_KEY`` is set, an HMAC-SHA256
signature is also recorded so the API can reject tampered artifacts before
deserializing (pickle/joblib payloads are not safe to load unchecked).

SEC-009: SHA-256 alone does not defend against a compromised Hub repo that replaces
both manifest and pickle; production requires ``CHECKPOINT_HMAC_KEY`` and HMAC
entries in the manifest, plus a pinned ``GAITGUARD_HF_REVISION``.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from pathlib import Path
from typing import Any

import joblib

MANIFEST_FILENAME = "checkpoint_manifest.json"
_MANIFEST_VERSION = 1
_FLOATING_HUB_REVISIONS = frozenset({"", "main", "master", "head"})


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint fails manifest or HMAC verification."""


def is_production_environment() -> bool:
    """True when API/training runs with ENVIRONMENT=production."""
    deploy_env = os.environ.get("ENVIRONMENT", os.environ.get("ENV", "development")).lower()
    return deploy_env in ("production", "prod")


def assert_production_checkpoint_policy() -> None:
    """
    SEC-009: production must configure HMAC before deserializing Hub checkpoints.

    SHA-256 manifest entries alone do not help if an attacker replaces manifest and
    pickle together; HMAC ties artifacts to a deployment secret.
    """
    if not is_production_environment():
        return
    if _hmac_key() is None:
        raise RuntimeError(
            "CHECKPOINT_HMAC_KEY must be set when ENVIRONMENT=production (SEC-009)."
        )


def is_floating_hub_revision(revision: str) -> bool:
    """True for branch names that move over time (unsafe for production pin)."""
    return revision.strip().lower() in _FLOATING_HUB_REVISIONS


def assert_production_hub_revision_policy(revision: str) -> None:
    """SEC-009: refuse floating Hub revisions in production downloads."""
    if not is_production_environment():
        return
    if is_floating_hub_revision(revision):
        raise RuntimeError(
            "GAITGUARD_HF_REVISION must be a pinned tag or commit hash in production "
            "(SEC-009); 'main' is not allowed."
        )


def _hmac_key() -> bytes | None:
    raw = os.environ.get("CHECKPOINT_HMAC_KEY")
    if not raw:
        return None
    return raw.encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_hex(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def manifest_path(directory: Path) -> Path:
    return Path(directory) / MANIFEST_FILENAME


def load_manifest(directory: Path) -> dict[str, Any]:
    path = manifest_path(directory)
    if not path.exists():
        return {"version": _MANIFEST_VERSION, "files": {}}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise CheckpointIntegrityError(f"Invalid manifest format: {path}")
    loaded.setdefault("version", _MANIFEST_VERSION)
    loaded.setdefault("files", {})
    return loaded


def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = manifest_path(directory)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def register_checkpoint_file(
    path: Path,
    *,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Hash (and optionally HMAC-sign) a checkpoint file and update the manifest."""
    path = Path(path)
    manifest_dir = manifest_dir or path.parent
    data = path.read_bytes()
    entry: dict[str, Any] = {
        "sha256": _sha256_hex(data),
        "size": len(data),
    }
    key = _hmac_key()
    if key is not None:
        entry["hmac"] = _hmac_hex(data, key)

    manifest = load_manifest(manifest_dir)
    manifest["version"] = _MANIFEST_VERSION
    files = manifest.setdefault("files", {})
    if not isinstance(files, dict):
        files = {}
        manifest["files"] = files
    files[path.name] = entry
    write_manifest(manifest_dir, manifest)
    return entry


def refresh_manifest(directory: Path, *, pattern: str = "*.pkl") -> dict[str, Any]:
    """Build or refresh manifest entries for all matching checkpoint files."""
    directory = Path(directory)
    for path in sorted(directory.glob(pattern)):
        if path.is_file():
            register_checkpoint_file(path, manifest_dir=directory)
    return load_manifest(directory)


def _verify_checkpoint_bytes(
    filename: str,
    data: bytes,
    manifest_dir: Path,
    *,
    require_manifest: bool,
    require_hmac: bool = False,
) -> None:
    manifest = load_manifest(manifest_dir)
    files = manifest.get("files", {})
    entry = files.get(filename) if isinstance(files, dict) else None

    if entry is None:
        if require_manifest:
            raise CheckpointIntegrityError(
                f"Checkpoint '{filename}' is not listed in {MANIFEST_FILENAME} "
                f"under {manifest_dir}"
            )
        return

    digest = _sha256_hex(data)
    if digest != entry.get("sha256"):
        raise CheckpointIntegrityError(
            f"Checkpoint '{filename}' failed SHA-256 verification "
            f"(manifest mismatch — file may be corrupted or tampered)"
        )

    expected_size = entry.get("size")
    if isinstance(expected_size, int) and len(data) != expected_size:
        raise CheckpointIntegrityError(
            f"Checkpoint '{filename}' size {len(data)} does not match manifest {expected_size}"
        )

    recorded_hmac = entry.get("hmac")
    must_verify_hmac = require_hmac or bool(recorded_hmac)
    if must_verify_hmac:
        key = _hmac_key()
        if key is None:
            raise CheckpointIntegrityError(
                f"Checkpoint '{filename}' requires CHECKPOINT_HMAC_KEY for verification"
            )
        if not recorded_hmac:
            raise CheckpointIntegrityError(
                f"Checkpoint '{filename}' has no HMAC signature in {MANIFEST_FILENAME}; "
                "re-sign checkpoints with CHECKPOINT_HMAC_KEY before production deploy (SEC-009)"
            )
        if not hmac.compare_digest(_hmac_hex(data, key), str(recorded_hmac)):
            raise CheckpointIntegrityError(
                f"Checkpoint '{filename}' failed HMAC verification"
            )


def save_checkpoint(
    path: Path,
    obj: Any,
    *,
    manifest_dir: Path | None = None,
) -> Path:
    """Persist an estimator with joblib and register it in the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    register_checkpoint_file(path, manifest_dir=manifest_dir or path.parent)
    return path


def load_checkpoint(
    path: Path,
    *,
    manifest_dir: Path | None = None,
    require_manifest: bool = False,
    require_hmac: bool | None = None,
) -> Any:
    """Load a checkpoint after optional manifest / HMAC verification."""
    path = Path(path)
    manifest_dir = manifest_dir or path.parent
    if require_hmac is None:
        require_hmac = is_production_environment()
    data = path.read_bytes()
    _verify_checkpoint_bytes(
        path.name,
        data,
        manifest_dir,
        require_manifest=require_manifest,
        require_hmac=require_hmac,
    )
    return joblib.load(io.BytesIO(data))
