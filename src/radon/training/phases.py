"""The two training phases.

*Warm-up* teaches the model to reconstruct. *Negative learning* keeps
reconstructing inliers while pushing suspected outliers towards their
feature-wise inverse, so that an outlier ends up badly reconstructed on purpose
and its anomaly score separates from the inlier population.

Both phases return an :class:`EpochStats` of raw sums; turning those into means
is the trainer's job, because only it knows how many samples the epoch saw.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from radon.config import ConsistencyConfig
from radon.models.autoencoder import RadonAutoencoder
from radon.training.projection import ProjectionState

__all__ = [
    "EpochStats",
    "gradient_norm",
    "negative_learning_epoch",
    "parameter_norm",
    "warmup_epoch",
]

logger = logging.getLogger(__name__)


@dataclass
class EpochStats:
    """Raw sums accumulated over one epoch.

    Attributes:
        n_seen: Samples that contributed to an optimisation step.
        loss_total: Sum of the total loss, weighted by batch size.
        loss_inlier: Sum of the inlier loss, weighted by batch size.
        outlier_loss_sum: Weighted sum of the outlier loss.
        outlier_weight_sum: Sum of the weights that normalise it.
        cos_outlier_self_sum: Sum of cosines between outlier reconstructions and
            the outliers themselves.
        cos_outlier_inverse_sum: Sum of cosines against the inverted target.
        n_outliers: Number of real outlier samples seen.
        grad_norm: Gradient norm after the final optimisation step.
        lambda_used: Weight applied to the outlier loss this epoch.
        nonfinite: Whether any batch was skipped for producing a non-finite loss.
    """

    n_seen: int = 0
    loss_total: float = 0.0
    loss_inlier: float = 0.0
    outlier_loss_sum: float = 0.0
    outlier_weight_sum: float = 0.0
    cos_outlier_self_sum: float = 0.0
    cos_outlier_inverse_sum: float = 0.0
    n_outliers: float = 0.0
    grad_norm: float = float("nan")
    lambda_used: float = 0.0
    nonfinite: bool = False


def gradient_norm(model: nn.Module) -> float:
    """L2 norm of all gradients currently attached to ``model``.

    Args:
        model: The model whose gradients to measure.

    Returns:
        The norm, or NaN if any gradient holds a non-finite value.

    Note:
        The trainer calls this once per epoch rather than once per batch. The
        original called it after every ``backward``, discarding all but the last
        result — the value it logged was always the final batch's. On GPU each
        call forces one host synchronisation per parameter tensor, so that was
        thousands of stalls per run for a number nobody read.
    """
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            return float("nan")
        total += gradient.pow(2).sum().item()
    return math.sqrt(total)


def parameter_norm(model: nn.Module) -> float:
    """L2 norm of all parameters.

    Args:
        model: The model to measure.

    Returns:
        The norm, or NaN if any parameter holds a non-finite value.
    """
    total = 0.0
    for parameter in model.parameters():
        value = parameter.detach()
        if not torch.isfinite(value).all():
            return float("nan")
        total += value.pow(2).sum().item()
    return math.sqrt(total)


def warmup_epoch(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.CosineSimilarity,
    has_outliers: bool,
) -> EpochStats:
    """Run one warm-up epoch.

    The objective pulls inlier reconstructions towards the inlier and pushes
    outlier reconstructions away from the outlier::

        (1 - cos(f(l), l))^2  +  cos(f(s), s)^2

    Args:
        model: The model to train.
        loader: Yields ``(inlier, outlier)`` pairs.
        optimizer: The optimiser.
        criterion: Cosine similarity over dimension 1.
        has_outliers: Whether the outlier half of each pair holds real data.
            When it does not, every ``s`` is a zero row, the outlier term is
            exactly zero and so is its gradient — the forward and backward
            passes over ``s`` are pure waste and are skipped. All reported
            quantities are unchanged; only the wasted compute disappears.

    Returns:
        The epoch's accumulated statistics.
    """
    model.train()
    stats = EpochStats()
    n_batches = len(loader)

    for index, (inliers, outliers) in enumerate(loader):
        optimizer.zero_grad()
        batch_size = inliers.size(0)

        inlier_loss_terms = (1.0 - criterion(model(inliers), inliers)) ** 2
        inlier_loss = inlier_loss_terms.mean()

        if has_outliers:
            outlier_cosines = criterion(model(outliers), outliers)
            outlier_loss_terms = outlier_cosines**2
            outlier_loss = outlier_loss_terms.mean()
            loss = inlier_loss + outlier_loss
        else:
            outlier_loss_terms = None
            outlier_cosines = None
            loss = inlier_loss

        if not torch.isfinite(loss).item():
            stats.nonfinite = True
            continue

        loss.backward()
        if index == n_batches - 1:
            stats.grad_norm = gradient_norm(model)
        optimizer.step()

        stats.n_seen += batch_size
        stats.loss_total += loss.item() * batch_size
        stats.loss_inlier += inlier_loss.item() * batch_size

        # The padded case contributes exact zeros; accumulating them explicitly
        # keeps the reported averages identical to the unoptimised path.
        if outlier_loss_terms is not None and outlier_cosines is not None:
            stats.outlier_loss_sum += outlier_loss_terms.sum().item()
            stats.cos_outlier_self_sum += outlier_cosines.sum().item()
        stats.outlier_weight_sum += float(batch_size)
        stats.n_outliers += float(batch_size)

    return stats


def negative_learning_epoch(
    model: RadonAutoencoder,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.CosineSimilarity,
    invert: Callable[[Tensor], Tensor],
    projection: ProjectionState,
    ramp_factor: float,
    numeric_mask: Tensor,
    consistency: ConsistencyConfig,
    device: torch.device,
) -> EpochStats:
    """Run one negative-learning epoch.

    Inliers are reconstructed as before. Real outliers are pushed towards
    ``invert(s)``, weighted by how far below the outlier threshold they sit, so
    that borderline samples influence the model less than confident ones.

    Args:
        model: The model to train.
        loader: Yields ``(inlier, outlier, mask)`` triples, where the mask marks
            which outlier rows are real rather than padding.
        optimizer: The optimiser.
        criterion: Cosine similarity over dimension 1.
        invert: Maps an outlier batch to its negative-learning target.
        projection: Current projection state, for the confidence weighting.
        ramp_factor: Scales the outlier weight while it warms in.
        numeric_mask: Columns eligible for consistency noise.
        consistency: Consistency-penalty settings.
        device: Device for scalar tensors.

    Returns:
        The epoch's accumulated statistics.
    """
    model.train()
    stats = EpochStats(lambda_used=ramp_factor * projection.lambda_eff)
    n_batches = len(loader)

    threshold = projection.threshold_outlier_tensor
    weight = ramp_factor * projection.lambda_eff

    for index, (inliers, outliers, mask) in enumerate(loader):
        optimizer.zero_grad()
        batch_size = inliers.size(0)

        inlier_loss = ((1.0 - criterion(model(inliers), inliers)) ** 2).mean()

        mask = mask.float()
        n_real_outliers = float(mask.sum().item())
        consistency_loss = torch.tensor(0.0, device=device)

        if n_real_outliers > 0:
            outlier_reconstruction = model(outliers)
            target = invert(outliers)

            outlier_loss_terms = 1.0 - criterion(outlier_reconstruction, target)
            cos_self = criterion(outlier_reconstruction, outliers)
            cos_inverse = criterion(outlier_reconstruction, target)

            weights, weight_sum = _confidence_weights(cos_self, mask, threshold)
            outlier_loss = (outlier_loss_terms * weights).sum() / weight_sum

            if consistency.weight > 0.0:
                consistency_loss = _consistency_penalty(
                    model,
                    outliers,
                    outlier_reconstruction,
                    numeric_mask,
                    consistency.sigma,
                    weights,
                    weight_sum,
                )

            stats.outlier_loss_sum += (outlier_loss_terms * weights).sum().item()
            stats.outlier_weight_sum += weight_sum.item()
            stats.cos_outlier_self_sum += (cos_self * mask).sum().item()
            stats.cos_outlier_inverse_sum += (cos_inverse * mask).sum().item()
            stats.n_outliers += n_real_outliers
        else:
            outlier_loss = torch.tensor(0.0, device=device)

        loss = inlier_loss + weight * outlier_loss
        if consistency.weight > 0.0:
            loss = loss + consistency.weight * consistency_loss

        if not torch.isfinite(loss).item():
            stats.nonfinite = True
            continue

        loss.backward()
        if index == n_batches - 1:
            stats.grad_norm = gradient_norm(model)
        optimizer.step()

        stats.n_seen += batch_size
        stats.loss_total += loss.item() * batch_size
        stats.loss_inlier += inlier_loss.item() * batch_size

    return stats


def _confidence_weights(
    cos_self: Tensor, mask: Tensor, threshold: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Weight each outlier by how confidently it sits below the threshold.

    Samples far below the outlier threshold get weight 1; those just under it
    get as little as 0.1. When no threshold is available, or when every sample
    happens to sit above it, the weighting falls back to the plain mask so the
    objective does not silently vanish.

    Args:
        cos_self: Cosine between each outlier and its reconstruction.
        mask: 1 for real outlier rows, 0 for padding.
        threshold: The outlier threshold as a scalar tensor, or ``None``.

    Returns:
        The per-sample weights and their sum, clamped away from zero.
    """
    weights = mask
    weight_sum = weights.sum().clamp(min=1e-12)

    if threshold is None:
        return weights, weight_sum

    confidence = (threshold - cos_self).clamp(min=0.0) * mask
    peak = confidence.max()
    if peak.item() <= 1e-12:
        return weights, weight_sum

    weights = (confidence / peak).clamp(min=0.1, max=1.0)
    return weights, weights.sum().clamp(min=1e-12)


def _consistency_penalty(
    model: RadonAutoencoder,
    outliers: Tensor,
    reconstruction: Tensor,
    numeric_mask: Tensor,
    sigma: float,
    weights: Tensor,
    weight_sum: Tensor,
) -> Tensor:
    """Penalise reconstructions that move under small input perturbations.

    Noise is added to numeric columns only; perturbing a binary indicator would
    change the category rather than jitter a magnitude.

    Args:
        model: The model.
        outliers: The outlier batch.
        reconstruction: Its unperturbed reconstruction.
        numeric_mask: Columns eligible for noise.
        sigma: Noise standard deviation.
        weights: Per-sample weights.
        weight_sum: Their sum.

    Returns:
        The weighted mean penalty.
    """
    perturbed = outliers.clone()
    if bool(numeric_mask.any()):
        noise = sigma * torch.randn_like(perturbed[:, numeric_mask])
        perturbed[:, numeric_mask] = (perturbed[:, numeric_mask] + noise).clamp(0.0, 1.0)

    perturbed_reconstruction = model(perturbed)
    per_sample = nn.functional.mse_loss(
        reconstruction, perturbed_reconstruction, reduction="none"
    ).mean(dim=1)
    return (per_sample * weights).sum() / weight_sum
