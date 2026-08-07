"""Image reconstruction plots for MNIST."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader

from radon.models.autoencoder import RadonAutoencoder

__all__ = ["ReconstructionSample", "collect_reconstructions", "plot_reconstructions"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionSample:
    """One image alongside its reconstruction.

    Attributes:
        original: The input image, already reshaped to 2-D.
        reconstruction: The model's output, same shape.
        score: Anomaly score, ``1 - cos``.
        label: Ground truth, 1 for an outlier.
    """

    original: Tensor
    reconstruction: Tensor
    score: float
    label: int

    @property
    def error_map(self) -> Tensor:
        """Absolute per-pixel reconstruction error."""
        return (self.original - self.reconstruction).abs()


@torch.no_grad()
def collect_reconstructions(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    image_shape: tuple[int, int],
    n_per_class: int = 3,
) -> tuple[list[ReconstructionSample], list[ReconstructionSample]]:
    """Gather a few inlier and outlier reconstructions.

    Stops as soon as enough of both classes have been seen, so it does not walk
    the whole split.

    Args:
        model: The trained model.
        loader: Yields ``(features, labels)`` batches.
        device: Device to run on.
        image_shape: Geometry to reshape the flattened vectors back into.
        n_per_class: How many samples to keep per class.

    Returns:
        The inlier and outlier samples.

    Raises:
        ValueError: If ``image_shape`` does not match the model's input width.
    """
    height, width = image_shape
    if height * width != model.input_dim:
        raise ValueError(
            f"collect_reconstructions: image_shape {image_shape} implies "
            f"{height * width} features but the model takes {model.input_dim}."
        )

    model.eval()
    model.to(device)
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)

    inliers: list[ReconstructionSample] = []
    outliers: list[ReconstructionSample] = []

    for features, labels in loader:
        features = features.to(device).float()
        reconstruction = model(features)
        scores = 1.0 - criterion(reconstruction, features)

        for index in range(features.shape[0]):
            label = int(labels[index].item())
            target = outliers if label == 1 else inliers
            if len(target) >= n_per_class:
                continue
            target.append(
                ReconstructionSample(
                    original=features[index].detach().cpu().reshape(height, width),
                    reconstruction=reconstruction[index].detach().cpu().reshape(height, width),
                    score=float(scores[index].item()),
                    label=label,
                )
            )

        if len(inliers) >= n_per_class and len(outliers) >= n_per_class:
            break

    logger.debug(
        "Collected %d inlier and %d outlier reconstruction(s).", len(inliers), len(outliers)
    )
    return inliers, outliers


def plot_reconstructions(samples: list[ReconstructionSample], title: str) -> Figure:
    """Plot originals, reconstructions and error maps in three columns.

    Args:
        samples: The samples to draw, one per row.
        title: Figure title.

    Returns:
        The figure.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    import matplotlib.pyplot as plt

    if not samples:
        raise ValueError("plot_reconstructions: no samples to plot.")

    fig, axes = plt.subplots(
        nrows=len(samples), ncols=3, figsize=(9, 2.6 * len(samples)), squeeze=False
    )
    fig.suptitle(title, fontsize=14)

    for row, sample in enumerate(samples):
        axes[row][0].imshow(sample.original, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][0].set_title(f"Input (label {sample.label})", fontsize=10)

        axes[row][1].imshow(sample.reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][1].set_title("Reconstruction", fontsize=10)

        image = axes[row][2].imshow(sample.error_map, cmap="hot")
        axes[row][2].set_title(f"|error|  (score {sample.score:.4f})", fontsize=10)
        fig.colorbar(image, ax=axes[row][2], fraction=0.046, pad=0.04)

        for column in range(3):
            axes[row][column].axis("off")

    fig.tight_layout()
    return fig
