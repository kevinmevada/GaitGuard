"""Central PyTorch device selection for all pipeline stages."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

_logged_device: torch.device | None = None


def _device_setting(config: dict[str, Any] | None) -> str:
    if config is None:
        return "auto"
    compute = config.get("compute") or {}
    if compute.get("device"):
        return str(compute["device"]).lower()
    dl = config.get("deep_learning") or {}
    return str(dl.get("device", "auto")).lower()


def resolve_torch_device(
    config: dict[str, Any] | None = None,
    *,
    log: bool = True,
) -> torch.device:
    """Resolve PyTorch device from ``compute.device`` or ``deep_learning.device``."""
    global _logged_device

    setting = _device_setting(config)
    if setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif setting.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "Requested device {!r} but CUDA is unavailable — using CPU",
                setting,
            )
            device = torch.device("cpu")
        else:
            device = torch.device(setting if ":" in setting else "cuda")
    else:
        device = torch.device("cpu")

    if log and _logged_device != device:
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device.index or 0)
            logger.info("PyTorch device: {} ({})", device, name)
        else:
            logger.info("PyTorch device: cpu")
        _logged_device = device

    return device
