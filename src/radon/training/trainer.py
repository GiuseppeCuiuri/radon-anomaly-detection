"""The RADON training loop.

Replaces the original ``fit``: one function of twenty-five parameters that also
owned data loading, projection, early stopping, timing and log formatting. Those
concerns now live in their own modules and this class only sequences them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from radon.config import CompatConfig, TrainingConfig
from radon.features.tabular import analyze_features, make_feature_inverter
from radon.models.autoencoder import RadonAutoencoder
from radon.training.early_stopping import EarlyStopper, reconstruction_loss
from radon.training.phases import (
    EpochStats,
    negative_learning_epoch,
    parameter_norm,
    warmup_epoch,
)
from radon.training.projection import Projector, pad_outliers_to_inliers
from radon.training.records import EpochRecord, ProjectionRecord, TrainingHistory

__all__ = ["RadonTrainer"]

logger = logging.getLogger(__name__)

WARMUP = "warmup"
NEGATIVE_LEARNING = "negative_learning"


@dataclass
class RadonTrainer:
    """Trains a :class:`RadonAutoencoder` with the RADON schedule.

    Attributes:
        model: The model to train, already on ``device``.
        cfg: The training schedule.
        device: Device the data and model live on.
        compat: Legacy-behaviour switches.
        score_batch_size: Rows per forward pass when scoring for a projection;
            ``0`` scores in a single pass.
    """

    model: RadonAutoencoder
    cfg: TrainingConfig
    device: torch.device
    compat: CompatConfig = field(default_factory=CompatConfig)
    score_batch_size: int = 0

    def __post_init__(self) -> None:
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-5)
        self.projector = Projector(cfg=self.cfg, device=self.device)
        self.stopper = EarlyStopper(
            cfg=self.cfg.early_stopping, warmup_epochs=self.cfg.warmup_epochs
        )
        self.history = TrainingHistory()
        self._frozen_inliers: Tensor | None = None

    def train(self, x: Tensor, feature_names: tuple[str, ...] | None = None) -> TrainingHistory:
        """Run the full schedule.

        Args:
            x: Training matrix of shape ``(N, D)``, values in ``[0, 1]``.
            feature_names: Column names, used to recognise one-hot groups.

        Returns:
            The complete telemetry of the run.

        Raises:
            ValueError: If ``x`` does not match the model's input width.
        """
        x = self._prepare_input(x)

        masks = analyze_features(x, feature_names=feature_names)
        invert = make_feature_inverter(masks.numeric)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        state = self.projector.state
        state.inliers = x
        state.outliers = None
        state.outlier_count = 0
        state.has_outliers = False
        state.lambda_eff = float(self.cfg.lambda_s)
        state.lambda_eff_ema = float(self.cfg.lambda_s)

        loader: DataLoader[Any] = self._warmup_loader(x, outliers=None)
        self._log_header(x)

        for epoch in range(1, self.cfg.epochs + 1):
            started = time.perf_counter()
            in_warmup = epoch <= self.cfg.warmup_epochs

            if epoch == self.cfg.warmup_epochs + 1:
                loader = self._enter_negative_learning()

            if in_warmup:
                stats = warmup_epoch(
                    self.model,
                    cast("DataLoader[tuple[Tensor, Tensor]]", loader),
                    optimizer,
                    self.criterion,
                    has_outliers=state.has_outliers,
                )
            else:
                stats = negative_learning_epoch(
                    self.model,
                    cast("DataLoader[tuple[Tensor, Tensor, Tensor]]", loader),
                    optimizer,
                    self.criterion,
                    invert=invert,
                    projection=state,
                    ramp_factor=self._ramp_factor(epoch),
                    numeric_mask=masks.numeric,
                    consistency=self.cfg.consistency,
                    device=self.device,
                )

            should_project = self._should_project(epoch)
            stop = False

            if should_project:
                record = self.projector.project(
                    self.model,
                    x,
                    epoch,
                    self.criterion,
                    score_batch_size=self.score_batch_size,
                )
                self.history.projections.append(record)
                self._log_projection(record)

                loader = self._rebuild_loader(in_warmup)

                if self.cfg.early_stopping.enabled:
                    stop = self.stopper.update(
                        metric=self._early_stopping_metric(x, stats),
                        epoch=epoch,
                        model=self.model,
                        projection_snapshot=state.snapshot(),
                    )

            duration = time.perf_counter() - started
            epoch_record = self._build_epoch_record(
                epoch=epoch,
                phase=WARMUP if in_warmup else NEGATIVE_LEARNING,
                stats=stats,
                duration=duration,
                had_projection=should_project,
            )
            self.history.epochs.append(epoch_record)
            logger.info("%s", epoch_record.format_line())

            if stop:
                self.history.stopped_early = True
                break

        self._finalise()
        return self.history

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _prepare_input(self, x: Tensor) -> Tensor:
        """Validate and move the training matrix onto the training device."""
        if x.ndim != 2:
            raise ValueError(
                f"RadonTrainer: the training matrix must be 2-D, got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.model.input_dim:
            raise ValueError(
                f"RadonTrainer: the model expects {self.model.input_dim} features but the "
                f"training matrix has {x.shape[1]}."
            )
        return x.to(self.device).float()

    def _log_header(self, x: Tensor) -> None:
        """Log the run's configuration once, up front."""
        logger.info(
            "Training on %d samples of dimension %d for %d epochs "
            "(%d warm-up), lr=%g, batch=%d, estimator=%s.",
            x.shape[0],
            x.shape[1],
            self.cfg.epochs,
            self.cfg.warmup_epochs,
            self.cfg.lr,
            self.cfg.batch_size,
            self.cfg.bin_estimator,
        )
        if self.compat.any_enabled:
            logger.warning(
                "Legacy compatibility flags active: %s. Results reproduce the original "
                "implementation, including its defects.",
                ", ".join(self.compat.enabled_flags()),
            )

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    def _warmup_loader(
        self, inliers: Tensor, outliers: Tensor | None
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build the warm-up loader, pairing each inlier with an outlier row.

        Shuffling is on unless the legacy flag restores the original
        ``shuffle=False``, which froze both the batch order and the inlier /
        outlier pairing for every epoch of the phase.

        Args:
            inliers: The inlier set.
            outliers: The outlier set, or ``None`` when none exist yet.

        Returns:
            A loader over ``(inlier, outlier)`` pairs.
        """
        padded, _ = pad_outliers_to_inliers(
            outliers, inliers.shape[0], inliers.shape[1], self.device
        )
        dataset = cast("Dataset[tuple[Tensor, Tensor]]", TensorDataset(inliers, padded))
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=not self.compat.legacy_warmup_no_shuffle,
            drop_last=False,
        )

    def _negative_learning_loader(
        self, inliers: Tensor, outliers: Tensor | None
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        """Build the negative-learning loader with its padding mask.

        Args:
            inliers: The inlier set.
            outliers: The outlier set, or ``None``.

        Returns:
            A loader over ``(inlier, outlier, mask)`` triples.
        """
        padded, mask = pad_outliers_to_inliers(
            outliers, inliers.shape[0], inliers.shape[1], self.device
        )
        dataset = cast(
            "Dataset[tuple[Tensor, Tensor, Tensor]]", TensorDataset(inliers, padded, mask)
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _enter_negative_learning(self) -> DataLoader[Any]:
        """Switch to the negative-learning regime at the phase boundary."""
        state = self.projector.state
        state.lambda_card_ema = None

        inliers = state.inliers
        assert inliers is not None  # set before the loop and at every projection

        if self.cfg.early_stopping.metric == "frozen_l0":
            self._frozen_inliers = inliers.clone()
            logger.info(
                "Negative learning begins; froze L0 with %d inliers as the early-stopping "
                "reference.",
                inliers.shape[0],
            )

        return self._negative_learning_loader(inliers, state.outliers)

    def _rebuild_loader(self, in_warmup: bool) -> DataLoader[Any]:
        """Rebuild the loader from the freshly computed partition."""
        state = self.projector.state
        inliers = state.inliers
        assert inliers is not None  # a projection always assigns it

        if in_warmup:
            return self._warmup_loader(inliers, state.outliers)
        return self._negative_learning_loader(
            inliers, state.outliers if state.has_outliers else None
        )

    # ------------------------------------------------------------------ #
    # Schedule
    # ------------------------------------------------------------------ #

    def _should_project(self, epoch: int) -> bool:
        """Whether a projection runs at the end of ``epoch``.

        A projection is forced at the last warm-up epoch so that negative
        learning always starts from a fresh partition rather than a stale one.

        Args:
            epoch: One-based epoch number.

        Returns:
            True when a projection should run.
        """
        warmup_epochs = self.cfg.warmup_epochs

        if epoch == warmup_epochs:
            return True
        if epoch < warmup_epochs:
            return epoch % self.cfg.projection_every_warmup == 0
        return (epoch - warmup_epochs) % self.cfg.projection_every_nl == 0

    def _ramp_factor(self, epoch: int) -> float:
        """Warm-in factor for the negative-learning weight.

        Args:
            epoch: One-based epoch number.

        Returns:
            A factor in ``[0, 1]``.
        """
        if not self.cfg.ramp.enabled:
            return 1.0
        if epoch <= self.cfg.warmup_epochs:
            return 0.0
        progress = (epoch - self.cfg.warmup_epochs) / max(self.cfg.ramp.epochs, 1)
        return min(1.0, progress)

    def _early_stopping_metric(self, x: Tensor, stats: EpochStats) -> float:
        """Compute the value early stopping monitors.

        Args:
            x: The full training matrix.
            stats: This epoch's statistics, used by the legacy metric.

        Returns:
            The monitored value, lower being better.
        """
        metric = self.cfg.early_stopping.metric
        if self.compat.legacy_early_stop_metric:
            metric = "current_l"

        if metric == "current_l":
            # The epoch's mean inlier loss, measured over whatever L was in
            # force while that epoch ran.
            if stats.n_seen == 0 or stats.nonfinite:
                return float("inf")
            return stats.loss_inlier / stats.n_seen

        if metric == "full_x":
            return reconstruction_loss(self.model, x, self.criterion)

        reference = self._frozen_inliers
        if reference is None:
            # Still in warm-up: L0 has not been captured yet, so fall back to
            # the current inlier set, which is the only thing that exists.
            inliers = self.projector.state.inliers
            reference = inliers if inliers is not None else x
        return reconstruction_loss(self.model, reference, self.criterion)

    # ------------------------------------------------------------------ #
    # Records
    # ------------------------------------------------------------------ #

    def _build_epoch_record(
        self,
        epoch: int,
        phase: str,
        stats: EpochStats,
        duration: float,
        had_projection: bool,
    ) -> EpochRecord:
        """Turn an epoch's raw sums into a record."""
        if stats.n_seen == 0 or stats.nonfinite:
            loss_mean = float("inf")
            inlier_mean = float("inf")
        else:
            loss_mean = stats.loss_total / stats.n_seen
            inlier_mean = stats.loss_inlier / stats.n_seen

        if stats.outlier_weight_sum > 0:
            outlier_mean = stats.outlier_loss_sum / max(stats.outlier_weight_sum, 1e-12)
            denominator = max(stats.n_outliers, 1.0)
            cos_self = stats.cos_outlier_self_sum / denominator
            cos_inverse = stats.cos_outlier_inverse_sum / denominator
        else:
            outlier_mean = 0.0
            cos_self = 0.0
            cos_inverse = float("nan")

        ratio = (
            (stats.lambda_used * outlier_mean) / (inlier_mean + 1e-9)
            if inlier_mean != float("inf")
            else float("nan")
        )

        return EpochRecord(
            epoch=epoch,
            phase=phase,
            loss_mean=loss_mean,
            loss_inlier_mean=inlier_mean,
            loss_outlier_mean=outlier_mean,
            lambda_used=stats.lambda_used,
            loss_ratio=ratio,
            n_outliers_seen=int(stats.n_outliers),
            cos_outlier_self=cos_self,
            cos_outlier_inverse=cos_inverse,
            grad_norm=stats.grad_norm,
            param_norm=parameter_norm(self.model),
            duration_s=duration,
            had_projection=had_projection,
            nonfinite=stats.nonfinite,
        )

    @staticmethod
    def _log_projection(record: ProjectionRecord) -> None:
        """Emit a projection's summary block."""
        for line in record.format_lines():
            logger.info("%s", line)

    def _finalise(self) -> None:
        """Restore the best weights and the projection state that produced them."""
        if not self.cfg.early_stopping.enabled:
            self._log_summary()
            return

        if self.stopper.restore_best(self.model, self.device):
            self.history.best_epoch = self.stopper.best_epoch
            self.history.best_metric = self.stopper.best_metric
            if self.stopper.best_snapshot is not None:
                self.projector.state.restore(self.stopper.best_snapshot, self.device)
                # The restored thresholds are the ones that belong to the model
                # being returned, so they close the threshold history.
                self._append_restored_projection()

        self._log_summary()

    def _append_restored_projection(self) -> None:
        """Record the thresholds reinstated alongside the restored weights."""
        if not self.history.projections:
            return

        state = self.projector.state
        template = self.history.projections[-1]
        if (
            template.threshold_inlier == state.threshold_inlier
            and template.threshold_outlier == state.threshold_outlier
        ):
            return

        self.history.projections.append(
            replace(
                template,
                epoch=self.stopper.best_epoch or template.epoch,
                threshold_inlier=state.threshold_inlier,
                threshold_outlier=state.threshold_outlier,
                lambda_eff=state.lambda_eff,
            )
        )

    def _log_summary(self) -> None:
        """Log wall-clock totals at the end of a run."""
        history = self.history
        if not history.epochs:
            return

        durations = [record.duration_s for record in history.epochs]
        with_projection = [r.duration_s for r in history.epochs if r.had_projection]
        without_projection = [r.duration_s for r in history.epochs if not r.had_projection]

        logger.info(
            "Finished %d epoch(s) in %.2fs (mean %.4fs). %d projection(s).",
            len(durations),
            history.total_duration_s,
            history.total_duration_s / len(durations),
            len(history.projections),
        )
        if with_projection and without_projection:
            overhead = sum(with_projection) / len(with_projection) - sum(without_projection) / len(
                without_projection
            )
            logger.info("Estimated projection overhead: %.4fs per epoch.", overhead)
