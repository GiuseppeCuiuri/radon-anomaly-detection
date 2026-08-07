"""Compute-device resolution."""

from __future__ import annotations

import logging

import torch

from radon.exceptions import ConfigError

__all__ = ["resolve_device"]

logger = logging.getLogger(__name__)


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device string from the config into a :class:`torch.device`.

    Args:
        spec: ``auto`` picks CUDA when available and falls back to CPU. Any
            other value (``cpu``, ``cuda``, ``cuda:1``) is honoured literally.

    Returns:
        The resolved device.

    Raises:
        ConfigError: If CUDA is requested explicitly but is unavailable, or the
            device string is not parseable. An explicit request is treated as an
            error rather than silently downgraded to CPU: a run that quietly
            takes twenty times longer is worse than one that fails immediately.
    """
    normalised = spec.strip().lower()

    if normalised == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info("Device: %s (%s).", device, torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.info("Device: cpu (CUDA unavailable).")
        return device

    try:
        device = torch.device(normalised)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(f"device: '{spec}' is not a valid device string: {exc}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            f"device: '{spec}' was requested but CUDA is unavailable. "
            "Use device: auto to fall back to CPU automatically."
        )

    logger.info("Device: %s.", device)
    return device
