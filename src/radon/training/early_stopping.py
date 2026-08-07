"""Early stopping across projections.

The metric is evaluated at each projection rather than each epoch, because a
projection is what changes the problem the model is solving.

Which inlier set to measure on is a real methodological choice, not an
implementation detail. The original monitored the reconstruction loss over the
*current* ``L`` — but ``L`` is rebuilt at every projection, so the quantity is
not comparable with the one before it. An ``L`` that has shrunk towards the
easiest samples shows a lower loss whether or not the model improved, which can
both mask real degradation and trigger a stop that reflects nothing but a
changed denominator. ``frozen_l0`` measures instead on the inlier set captured
when negative learning began, held fixed thereafter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from radon.config import EarlyStoppingConfig
from radon.models.autoencoder import RadonAutoencoder

__all__ = ["EarlyStopper"]

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopper:
    """Tracks the best model seen and decides when to stop.

    Warm-up and negative learning are treated as separate regimes: the best
    warm-up model is tracked, then the counter resets when negative learning
    starts, because the objective changes and the two losses are not on the same
    scale.

    Attributes:
        cfg: Stopping configuration.
        warmup_epochs: Number of warm-up epochs, used to tell the regimes apart.
        best_metric: Best value seen in the current regime.
        best_epoch: Epoch that produced it.
        best_state: CPU copy of the weights at that epoch.
        best_snapshot: Projection state captured alongside those weights.
        strikes: Consecutive projections without improvement.
    """

    cfg: EarlyStoppingConfig
    warmup_epochs: int

    best_metric: float = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    best_snapshot: dict[str, Any] | None = None
    strikes: int = 0
    _negative_learning_baseline_set: bool = field(default=False, repr=False)

    def update(
        self,
        metric: float,
        epoch: int,
        model: RadonAutoencoder,
        projection_snapshot: dict[str, Any],
    ) -> bool:
        """Record a projection's metric and report whether to stop.

        Args:
            metric: The monitored value, lower being better.
            epoch: Epoch at which the projection ran.
            model: The model to snapshot when this is the best result so far.
            projection_snapshot: Projection state to restore alongside it.

        Returns:
            True when the patience has run out and training should stop.
        """
        if not np.isfinite(metric):
            logger.warning(
                "Early stopping skipped at epoch %d: the monitored metric is not finite, "
                "which means every batch this epoch produced a non-finite loss.",
                epoch,
            )
            return False

        if epoch <= self.warmup_epochs:
            self._update_warmup(metric, epoch, model, projection_snapshot)
            return False

        return self._update_negative_learning(metric, epoch, model, projection_snapshot)

    def _update_warmup(
        self,
        metric: float,
        epoch: int,
        model: RadonAutoencoder,
        projection_snapshot: dict[str, Any],
    ) -> None:
        """Track the best warm-up model without ever stopping."""
        improved = self.best_state is None or metric < (self.best_metric - self.cfg.min_delta)
        if improved:
            self._capture(metric, epoch, model, projection_snapshot)
            logger.debug("Best warm-up model: %.6f at epoch %d.", metric, epoch)

        self.strikes = 0
        self._negative_learning_baseline_set = False

    def _update_negative_learning(
        self,
        metric: float,
        epoch: int,
        model: RadonAutoencoder,
        projection_snapshot: dict[str, Any],
    ) -> bool:
        """Track the best negative-learning model and count strikes."""
        if not self._negative_learning_baseline_set:
            # The objective just changed; the warm-up best is not comparable.
            self._capture(metric, epoch, model, projection_snapshot)
            self.strikes = 0
            self._negative_learning_baseline_set = True
            logger.debug("Negative-learning baseline set at epoch %d: %.6f.", epoch, metric)
            return False

        if metric < (self.best_metric - self.cfg.min_delta):
            self._capture(metric, epoch, model, projection_snapshot)
            self.strikes = 0
            return False

        self.strikes += 1
        logger.debug(
            "No improvement at epoch %d (%.6f vs best %.6f): strike %d of %d.",
            epoch,
            metric,
            self.best_metric,
            self.strikes,
            self.cfg.patience,
        )

        if self.strikes >= self.cfg.patience:
            logger.info(
                "Early stopping at epoch %d after %d projections without improvement; "
                "restoring epoch %s (metric %.6f).",
                epoch,
                self.strikes,
                self.best_epoch,
                self.best_metric,
            )
            return True

        return False

    def _capture(
        self,
        metric: float,
        epoch: int,
        model: RadonAutoencoder,
        projection_snapshot: dict[str, Any],
    ) -> None:
        """Store the model weights and projection state as the new best."""
        self.best_metric = metric
        self.best_epoch = epoch
        self.best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        self.best_snapshot = dict(projection_snapshot)

    def restore_best(self, model: RadonAutoencoder, device: torch.device) -> bool:
        """Load the best weights back into ``model``.

        Args:
            model: The model to restore into.
            device: Device to move it to afterwards.

        Returns:
            True if weights were restored, False if none were ever captured.
        """
        if self.best_state is None:
            return False

        model.load_state_dict(self.best_state, strict=True)
        model.to(device)
        logger.info(
            "Restored the best model from epoch %s (metric %.6f).",
            self.best_epoch,
            self.best_metric,
        )
        return True


@torch.no_grad()
def reconstruction_loss(
    model: RadonAutoencoder, x: Tensor, criterion: nn.CosineSimilarity
) -> float:
    """Mean squared cosine reconstruction loss over a fixed set.

    Matches the inlier term of the training objective, so the monitored value is
    on the same scale as the loss being optimised.

    Args:
        model: The model to evaluate.
        x: The set to evaluate on.
        criterion: Cosine similarity over dimension 1.

    Returns:
        The mean loss, or infinity for an empty set.
    """
    if x.numel() == 0:
        return float("inf")

    was_training = model.training
    model.eval()
    try:
        loss = ((1.0 - criterion(model(x), x)) ** 2).mean()
    finally:
        model.train(was_training)

    return float(loss.item())
