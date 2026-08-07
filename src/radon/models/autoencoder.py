"""The RADON autoencoder.

A symmetric fully-connected autoencoder with a sigmoid output, which is why
every feature must be scaled to ``[0, 1]``: the decoder cannot represent
anything outside that range.

The architecture is unchanged from the reference implementation - four linear
layers, ReLU after all but the last, sigmoid on the output, and PyTorch's
default initialisation. What is new is that the model carries its own
dimensions, so downstream code no longer has to be told ``input_dim`` and
``784`` separately.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from radon.config import ModelConfig

__all__ = ["RadonAutoencoder", "build_model"]

logger = logging.getLogger(__name__)


class RadonAutoencoder(nn.Module):
    """Fully-connected autoencoder used by RADON.

    The forward pass is ``decode(encode(x))``:

    - encoder: ``input_dim -> hidden_dim -> latent_dim``, ReLU after each layer
    - decoder: ``latent_dim -> hidden_dim -> input_dim``, ReLU then sigmoid

    Attributes:
        input_dim: Number of input features.
        hidden_dim: Width of the outer hidden layers (``dim1`` originally).
        latent_dim: Bottleneck width (``dim2`` originally).
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Build the network.

        Args:
            input_dim: Number of input features.
            hidden_dim: Width of the outer hidden layers.
            latent_dim: Bottleneck width.

        Raises:
            ValueError: If any dimension is not positive.
        """
        super().__init__()

        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("latent_dim", latent_dim),
        ):
            if value <= 0:
                raise ValueError(f"RadonAutoencoder: {name} must be positive, got {value}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        # decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor) -> Tensor:
        """Map a batch of inputs to the latent space.

        Args:
            x: Batch of shape ``(N, input_dim)``.

        Returns:
            Latent codes of shape ``(N, latent_dim)``.
        """
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))

    def decode(self, z: Tensor) -> Tensor:
        """Map latent codes back to the input space.

        Args:
            z: Latent codes of shape ``(N, latent_dim)``.

        Returns:
            Reconstructions of shape ``(N, input_dim)``, each value in
            ``(0, 1)`` because of the sigmoid output.
        """
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a batch of inputs.

        Args:
            x: Batch of shape ``(N, input_dim)``.

        Returns:
            Reconstructions of shape ``(N, input_dim)``.

        Raises:
            ValueError: If ``x`` is not 2-D or its width is not ``input_dim``.
                The original code reshaped silently at each call site, which
                hid dimension mistakes until they surfaced as bad scores.
        """
        if x.ndim != 2:
            raise ValueError(
                f"RadonAutoencoder: expected a 2-D batch (N, {self.input_dim}), "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"RadonAutoencoder: expected {self.input_dim} features, got {x.shape[1]}."
            )
        return self.decode(self.encode(x))

    @property
    def architecture(self) -> dict[str, int]:
        """The three dimensions, as stored in checkpoints."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the model's parameters.

        Args:
            trainable_only: Count only parameters that require gradients.

        Returns:
            The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def extra_repr(self) -> str:
        """Dimensions shown by ``print(model)``."""
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"latent_dim={self.latent_dim}"
        )


def build_model(cfg: ModelConfig, input_dim: int) -> RadonAutoencoder:
    """Instantiate the autoencoder for a dataset of the given width.

    Args:
        cfg: Architecture configuration.
        input_dim: Number of features in the dataset.

    Returns:
        An initialised model on the CPU; move it with ``.to(device)``.

    Raises:
        ValueError: If ``input_dim`` is not positive.
    """
    model = RadonAutoencoder(
        input_dim=input_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim
    )

    if cfg.latent_dim >= input_dim:
        logger.warning(
            "latent_dim (%d) is not smaller than input_dim (%d): the autoencoder can learn "
            "the identity map, so reconstruction error stops separating inliers from "
            "outliers. Reduce model.latent_dim.",
            cfg.latent_dim,
            input_dim,
        )

    logger.info(
        "Model: %d -> %d -> %d -> %d -> %d (%s trainable parameters).",
        input_dim,
        cfg.hidden_dim,
        cfg.latent_dim,
        cfg.hidden_dim,
        input_dim,
        f"{model.num_parameters():,}",
    )
    return model
