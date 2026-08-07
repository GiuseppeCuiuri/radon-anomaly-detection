"""Checkpoint persistence.

The original code saved a bare ``state_dict``. That file cannot be reloaded
without independently remembering the three dimensions it was built with, and
nothing recorded them - the widths lived in a notebook cell that also encoded
them as an inline rule on ``input_dim``. A checkpoint here therefore stores the
architecture alongside the weights, plus an ``extra`` slot for run-specific
values such as decision thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from radon import __version__
from radon.exceptions import CheckpointError
from radon.models.autoencoder import RadonAutoencoder

__all__ = ["CHECKPOINT_FORMAT", "LoadedCheckpoint", "load_checkpoint", "save_checkpoint"]

logger = logging.getLogger(__name__)

#: Bumped whenever the on-disk layout changes incompatibly.
CHECKPOINT_FORMAT = 1

_REQUIRED_KEYS = ("format", "architecture", "state_dict")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A restored checkpoint.

    Attributes:
        model: The reconstructed model, with weights loaded.
        extra: Whatever the caller stored at save time, e.g. thresholds.
        radon_version: Package version that produced the file.
    """

    model: RadonAutoencoder
    extra: dict[str, Any] = field(default_factory=dict)
    radon_version: str = "unknown"


def save_checkpoint(
    model: RadonAutoencoder,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a model and its architecture to disk.

    Args:
        model: The model to persist.
        path: Destination file. Parent directories are created as needed.
        extra: Run-specific values to store alongside the weights. Restrict
            these to tensors and plain scalars, lists and dicts: checkpoints are
            loaded with ``weights_only=True``, which refuses arbitrary pickled
            objects such as NumPy arrays.

    Returns:
        The path that was written.

    Raises:
        CheckpointError: If the file cannot be written.
    """
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "radon_version": __version__,
        "architecture": model.architecture,
        "state_dict": model.state_dict(),
        "extra": dict(extra or {}),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    except OSError as exc:
        raise CheckpointError(f"Cannot write the checkpoint to '{path}': {exc}") from exc

    logger.info("Checkpoint saved to '%s'.", path)
    return path


def load_checkpoint(path: Path, device: torch.device | str | None = None) -> LoadedCheckpoint:
    """Restore a model saved by :func:`save_checkpoint`.

    The architecture comes from the file, so the caller does not have to know
    how the model was built.

    Args:
        path: Checkpoint file.
        device: Device to map the weights onto. ``None`` keeps them on the CPU.

    Returns:
        The restored model together with any extra payload.

    Raises:
        CheckpointError: If the file is missing, unreadable, written by an
            incompatible format version, or missing required fields.
    """
    map_location = torch.device(device) if device is not None else torch.device("cpu")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError as exc:
        raise CheckpointError(f"Checkpoint not found: '{path}'.") from exc
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"Cannot read the checkpoint '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError(
            f"'{path}' does not hold a RADON checkpoint (found {type(payload).__name__}). "
            "A bare state_dict from the original code needs its dimensions supplied by hand: "
            "build a RadonAutoencoder explicitly and call load_state_dict on it."
        )

    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"'{path}' is missing the field(s) {missing}.")

    stored_format = payload["format"]
    if stored_format != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"'{path}' uses checkpoint format {stored_format}, but this version of RADON "
            f"reads format {CHECKPOINT_FORMAT}."
        )

    architecture = payload["architecture"]
    try:
        model = RadonAutoencoder(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            latent_dim=int(architecture["latent_dim"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"'{path}' has an unusable architecture record {architecture!r}: {exc}"
        ) from exc

    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"'{path}': the stored weights do not match the stored architecture "
            f"{architecture}: {exc}"
        ) from exc

    model.to(map_location)

    logger.info("Checkpoint loaded from '%s' (%s).", path, model.extra_repr())
    return LoadedCheckpoint(
        model=model,
        extra=dict(payload.get("extra", {})),
        radon_version=str(payload.get("radon_version", "unknown")),
    )
