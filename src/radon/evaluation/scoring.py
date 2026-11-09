"""Anomaly scoring on a held-out split.

The anomaly score is ``1 - cos(f(x), x)``: zero for a perfectly reconstructed
sample and larger the worse the reconstruction. It is the same quantity the
projection step thresholds during training, so a training threshold transfers
directly to test time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from radon.models.autoencoder import RadonAutoencoder

__all__ = ["ScoredSplit", "anomaly_scores", "score_split"]

logger = logging.getLogger(__name__)


def anomaly_scores(model: RadonAutoencoder, features: torch.Tensor) -> torch.Tensor:
    """Anomaly score of each row: ``1 - cos(f(x), x)``.

    Args:
        model: The trained model.
        features: Batch of shape ``(N, input_dim)``, on the model's device.

    Returns:
        One score per row, shape ``(N,)``. Gradients follow the caller's
        context, so wrap the call in ``torch.no_grad()`` for inference.
    """
    criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
    scores: torch.Tensor = 1.0 - criterion(model(features), features)
    return scores


@dataclass(frozen=True)
class ScoredSplit:
    """Anomaly scores and ground truth for one evaluation split.

    Attributes:
        scores: Anomaly score per sample, ``1 - cos(f(x), x)``.
        labels: Binary ground truth, 1 for an outlier.
        predictions: Binary decisions when a threshold was supplied, else
            ``None``.
        threshold: The anomaly-score threshold used, else ``None``.
    """

    scores: NDArray[np.float64]
    labels: NDArray[np.int64]
    predictions: NDArray[np.int64] | None = None
    threshold: float | None = None

    @property
    def n_samples(self) -> int:
        """Number of scored samples."""
        return int(self.scores.size)

    @property
    def anomaly_rate(self) -> float:
        """Fraction of samples labelled as outliers."""
        return float(self.labels.mean())

    def confusion(self) -> dict[str, int]:
        """Confusion counts against the ground truth.

        Returns:
            Counts keyed ``tp``, ``fp``, ``tn``, ``fn``.

        Raises:
            ValueError: If no threshold was applied, so there are no decisions
                to compare.
        """
        if self.predictions is None:
            raise ValueError(
                "ScoredSplit.confusion: no predictions available. Pass a threshold to "
                "score_split to obtain binary decisions."
            )
        predicted = self.predictions.astype(bool)
        actual = self.labels.astype(bool)
        return {
            "tp": int((predicted & actual).sum()),
            "fp": int((predicted & ~actual).sum()),
            "tn": int((~predicted & ~actual).sum()),
            "fn": int((~predicted & actual).sum()),
        }

    def describe(self) -> str:
        """One-line summary of the score distribution."""
        inliers = self.scores[self.labels == 0]
        outliers = self.scores[self.labels == 1]
        summary = (
            f"{self.n_samples} samples, anomaly rate {self.anomaly_rate:.4f} | "
            f"score in [{np.min(self.scores):.4f}, {np.max(self.scores):.4f}]"
        )

        if inliers.size == 0 or outliers.size == 0:
            present = "outliers" if inliers.size == 0 else "inliers"
            return f"{summary} | one class only ({present}), separation undefined"

        separation = "separated" if inliers.mean() < outliers.mean() else "overlapping"
        return (
            f"{summary} | inlier mean {inliers.mean():.4f} vs "
            f"outlier mean {outliers.mean():.4f} ({separation})"
        )


@torch.no_grad()
def score_split(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    threshold_cosine: float | None = None,
) -> ScoredSplit:
    """Score every sample in ``loader``.

    Args:
        model: The trained model.
        loader: Yields ``(features, labels)`` batches in a fixed order.
        device: Device to run on.
        threshold_cosine: A *cosine* threshold from training, as produced by the
            projection step. A sample is called an outlier when its cosine falls
            below it, equivalently when its anomaly score exceeds
            ``1 - threshold_cosine``. ``None`` skips the decision step.

    Returns:
        The scores, labels and optional decisions.

    Raises:
        ValueError: If the loader is empty.
    """
    model.eval()
    model.to(device)

    # Accumulate whole arrays rather than per-sample Python floats: the original
    # extended a list with one NumPy scalar per sample, allocating an object for
    # every point in the split.
    score_batches: list[NDArray[np.float64]] = []
    label_batches: list[NDArray[np.int64]] = []

    for features, labels in loader:
        features = features.to(device).float()
        batch_scores = anomaly_scores(model, features)

        score_batches.append(batch_scores.detach().cpu().numpy())
        label_batches.append(labels.detach().cpu().numpy())

    if not score_batches:
        raise ValueError("score_split: the loader produced no batches.")

    scores = np.concatenate(score_batches).astype(np.float64).reshape(-1)
    labels = np.concatenate(label_batches).astype(np.int64).reshape(-1)

    predictions: NDArray[np.int64] | None = None
    threshold: float | None = None
    if threshold_cosine is not None:
        threshold = 1.0 - float(threshold_cosine)
        predictions = (scores > threshold).astype(np.int64)

    result = ScoredSplit(scores=scores, labels=labels, predictions=predictions, threshold=threshold)
    logger.info("Scored split: %s", result.describe())
    return result
