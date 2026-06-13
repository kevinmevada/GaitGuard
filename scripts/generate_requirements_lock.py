#!/usr/bin/env python3
"""Write pinned requirements-lock.txt from the active environment (REP-001 / RES-007)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQ = REPO_ROOT / "fall_risk_pipeline" / "requirements.txt"
DEFAULT_OUT = REPO_ROOT / "fall_risk_pipeline" / "requirements-lock.txt"

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

HEADER_CPU = """# Auto-generated CPU lockfile (RES-007) — regenerate after dependency changes:
#   python scripts/generate_requirements_lock.py --install --torch-index cpu
#
# Install exact versions (CI / CPU researchers):
#   pip install -r fall_risk_pipeline/requirements-lock.txt
#
# GPU training (optional — reinstall torch from CUDA index if needed):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
# Or regenerate a CUDA lockfile:
#   python scripts/generate_requirements_lock.py --install --torch-index cu128 \\
#     --output fall_risk_pipeline/requirements-lock-cu128.txt

"""

HEADER_CUDA = """# Auto-generated CUDA lockfile (cu128) — optional GPU reproducibility:
#   python scripts/generate_requirements_lock.py --install --torch-index cu128 \\
#     --output fall_risk_pipeline/requirements-lock-cu128.txt
#
# Install:
#   pip install -r fall_risk_pipeline/requirements-lock-cu128.txt
#
# CPU-only researchers should use requirements-lock.txt instead.

"""

REQ_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_requirement_names(req_path: Path) -> list[str]:
    names: list[str] = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        token = line.split("[", 1)[0]
        for sep in (">=", "==", "<=", "!=", "~=", "<", ">"):
            if sep in token:
                token = token.split(sep, 1)[0]
                break
        token = token.strip()
        if REQ_NAME_RE.match(token):
            names.append(normalize_name(token))
    return names


def parse_freeze(pins_text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in pins_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        if "==" not in stripped:
            continue
        name, _version = stripped.split("==", 1)
        pins[normalize_name(name)] = stripped
    return pins


def pip_show_pin(name: str) -> str | None:
    """Resolve an installed package to name==version (works with conda + pip installs)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    fields: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    pkg_name = fields.get("name")
    version = fields.get("version")
    if not pkg_name or not version:
        return None
    return f"{pkg_name}=={version}"


def resolve_pin(name: str, pins: dict[str, str]) -> str | None:
    line = pins.get(name)
    if line is not None:
        return line
    # PyPI name vs import name fallbacks
    aliases = {
        "pywavelets": "PyWavelets",
        "pyyaml": "PyYAML",
        "scikit-learn": "scikit-learn",
    }
    lookup = aliases.get(name, name)
    return pip_show_pin(lookup)


def install_requirements(req_path: Path, *, torch_index: str) -> None:
    index = TORCH_CPU_INDEX if torch_index == "cpu" else TORCH_CUDA_INDEX
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "torch",
            "--index-url",
            index,
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
        check=True,
    )


def build_lock_lines(required_names: list[str], pins: dict[str, str]) -> tuple[list[str], list[str]]:
    wanted: list[str] = []
    missing: list[str] = []
    for name in required_names:
        line = resolve_pin(name, pins)
        if line is None:
            missing.append(name)
            continue
        wanted.append(line)
    return sorted(set(wanted)), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-index",
        choices=("cpu", "cu128"),
        default="cpu",
        help="PyTorch wheel index used when --install is set (default: cpu)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="pip install fall_risk_pipeline/requirements.txt before freezing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Lockfile output path",
    )
    args = parser.parse_args()

    if not REQ.is_file():
        print(f"Missing {REQ}", file=sys.stderr)
        return 1

    if args.install:
        print(f"Installing requirements with torch index: {args.torch_index}")
        install_requirements(REQ, torch_index=args.torch_index)

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    pins = parse_freeze(proc.stdout)
    required = parse_requirement_names(REQ)
    wanted, missing = build_lock_lines(required, pins)

    if missing:
        print(
            "Missing packages from active environment (install first):\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "Hint: python scripts/generate_requirements_lock.py --install --torch-index cpu",
            file=sys.stderr,
        )
        return 1

    header = HEADER_CUDA if args.torch_index == "cu128" else HEADER_CPU
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + "\n".join(wanted) + "\n", encoding="utf-8")
    rel = args.output.relative_to(REPO_ROOT)
    print(f"Wrote {rel} ({len(wanted)} packages, torch-index={args.torch_index})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
