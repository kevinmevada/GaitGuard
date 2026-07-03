#!/usr/bin/env python3
"""Stage per-shard inputs from OSDF onto the worker scratch directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

try:
    from local_paths import (
        gaitguard_root,
        local_copy,
        local_staging_enabled,
        osd_uri_to_local_path,
    )
except ImportError:  # OSPool worker without local_paths.py — OSDF path only.
    def local_staging_enabled() -> bool:
        return False

    def gaitguard_root():  # pragma: no cover
        return Path(".")

    def local_copy(*_a, **_k) -> bool:  # pragma: no cover
        return False

    def osd_uri_to_local_path(_uri):  # pragma: no cover
        return None


# Hard cap per stashcp transfer. Without this, a wedged OSDF cache hangs the
# job until OSPool kills it at the 20h execute-duration limit (hold code 47).
_STASHCP_TIMEOUT_S = int(os.environ.get("GAITGUARD_STASHCP_TIMEOUT", "900"))


def _osdf_copy(src: str, dst: Path, *, recursive: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    local_src = osd_uri_to_local_path(src)
    if local_src is not None:
        return local_copy(local_src, dst, recursive=recursive)
    flags = ["-r"] if recursive else []
    for url in (src, src if src.startswith("osdf://") else f"osdf://{src.lstrip('/')}"):
        try:
            subprocess.run(
                ["stashcp", *flags, url, str(dst)],
                check=True,
                timeout=_STASHCP_TIMEOUT_S,
            )
            return True
        except subprocess.TimeoutExpired:
            logger.warning("stashcp timed out after {}s: {} -> {}", _STASHCP_TIMEOUT_S, url, dst)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.warning("stashcp failed {} -> {}: {}", url, dst, exc)
    return False


def _stage_ingest(manifest: dict, osd_gg: str, raw_root: Path) -> int:
    paths = manifest.get("trial_source_paths") or {}
    ok = 0
    scratch = Path.cwd()
    for trial_id, rel in paths.items():
        rel = str(rel).strip("/")
        trial_dir = raw_root / rel
        if trial_dir.is_dir():
            ok += 1
            continue
        # HTCondor osdf:// transfer may land at cwd/<rel>, cwd/<basename>, etc.
        linked = False
        for cand in (scratch / rel, scratch / Path(rel).name):
            if cand.is_dir():
                trial_dir.parent.mkdir(parents=True, exist_ok=True)
                if not trial_dir.exists():
                    cand.rename(trial_dir)
                linked = True
                ok += 1
                break
        if linked:
            continue
        if local_staging_enabled():
            local_trial = gaitguard_root() / "raw" / rel
            if local_copy(local_trial, trial_dir, recursive=True):
                ok += 1
                continue
        src = f"{osd_gg}/raw/{rel}?recursive"
        if _osdf_copy(src, trial_dir, recursive=True):
            ok += 1
        else:
            logger.warning("ingest staging miss: trial {} ({})", trial_id, rel)
    logger.info("ingest staging: {}/{} trial dirs ready", ok, len(paths))
    if ok < len(paths):
        # Fail LOUDLY: a partial chunk would be packaged, uploaded, and then
        # counted as done by --skip-existing-ingest -> silent data loss.
        logger.error("ingest staging incomplete ({}/{}); failing job for retry", ok, len(paths))
        return 2
    return 0


_SENSORS = ("head", "lower_back", "left_foot", "right_foot")


def _link_transferred_file(scratch: Path, dst: Path, *names: str) -> bool:
    if dst.is_file():
        return True
    for name in names:
        cand = scratch / name
        if cand.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            cand.rename(dst)
            return True
    return False


def _stage_preprocess_or_features(
    manifest: dict, osd_gg: str, proc_root: Path, *, signals: bool
) -> None:
    scratch = Path.cwd()
    meta_local = proc_root / "trial_metadata.csv"
    if not meta_local.is_file():
        if not _link_transferred_file(scratch, meta_local, "trial_metadata.csv"):
            _osdf_copy(f"{osd_gg}/processed/trial_metadata.csv", meta_local)
    if not signals:
        return
    signals_dir = proc_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in manifest.get("trial_ids", []):
        tid = str(trial_id)
        for sensor in _SENSORS:
            name = f"{tid}_{sensor}.parquet"
            dst = signals_dir / name
            if dst.is_file():
                continue
            if not _link_transferred_file(scratch, dst, name):
                src = f"{osd_gg}/processed/signals/{name}"
                _osdf_copy(src, dst)


def _config_data_roots() -> tuple[Path, Path]:
    """Read paths.raw_data / paths.processed_data from the active config.

    Falls back to the OSPool defaults (data/raw, data/processed) when the
    config can't be read, so behavior on ap40 is unchanged.
    """
    raw = Path("data/raw")
    proc = Path("data/processed")
    cfg_path = os.environ.get("GAITGUARD_CONFIG")
    if cfg_path and Path(cfg_path).is_file():
        try:
            import yaml

            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            paths = cfg.get("paths") or {}
            raw = Path(paths.get("raw_data", raw))
            proc = Path(paths.get("processed_data", proc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not read config paths, using defaults: {}", exc)
    return raw, proc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ingest", "preprocess", "features"])
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("_CONDOR_SCRATCH_DIR"):
        return 0

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    osd_gg = os.environ.get("OSDF_GAITGUARD", "osdf:///ospool/ap40/data/kevin.mevada/gaitguard")
    # Stage into the SAME dirs the shard job reads from (config paths), so the
    # staging target and DataLoader input never drift apart.
    raw_root, proc_root = _config_data_roots()
    raw_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "ingest":
        return _stage_ingest(manifest, osd_gg, raw_root)
    elif args.stage == "preprocess":
        _stage_preprocess_or_features(manifest, osd_gg, proc_root, signals=True)
    else:
        _stage_preprocess_or_features(manifest, osd_gg, proc_root, signals=False)
        signals_clean = proc_root / "signals_clean"
        signals_clean.mkdir(parents=True, exist_ok=True)
        scratch = Path.cwd()
        for trial_id in manifest.get("trial_ids", []):
            tid = str(trial_id)
            for sensor in _SENSORS:
                name = f"{tid}_{sensor}.parquet"
                dst = signals_clean / name
                if dst.is_file():
                    continue
                if not _link_transferred_file(scratch, dst, name):
                    src = f"{osd_gg}/processed/signals_clean/{name}"
                    _osdf_copy(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
