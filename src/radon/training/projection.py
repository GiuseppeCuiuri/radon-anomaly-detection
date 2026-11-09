"""The projection step: partitioning the training set into inliers and outliers.

Every few epochs RADON scores the whole training matrix by cosine similarity
between each sample and its reconstruction, histograms those scores, and cuts
the distribution twice:

- above ``threshold_inlier`` a sample joins the inlier set ``L``;
- below ``threshold_outlier`` it joins the outlier candidate set ``S``;
- in between it is left unassigned and contributes to neither loss.

During warm-up the two thresholds coincide on the first projection, so the data
is split in two. Afterwards each threshold is estimated on its own half of the
histogram, the outlier threshold is smoothed across projections, and a
curriculum admits outliers gradually.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from radon.config import TrainingConfig
from radon.models.autoencoder import RadonAutoencoder
from radon.training.records import ProjectionRecord
from radon.training.thresholding import (
    UnimodalThreshold,
    histogram_bin_edges,
    unimodal_threshold,
)

__all__ = ["ProjectionState", "Projector", "score_dataset"]

logger = logging.getLogger(__name__)

#: Smallest gap enforced between the two thresholds, so that they cannot cross.
THRESHOLD_MARGIN = 1e-4

_PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


@torch.no_grad()
def score_dataset(
    model: RadonAutoencoder,
    x: Tensor,
    criterion: nn.CosineSimilarity,
    batch_size: int = 0,
) -> NDArray[np.floating]:
    """Cosine similarity between every sample and its reconstruction.

    Args:
        model: The model to score with. Left in whatever mode it was in; the
            caller is responsible for ``eval()``.
        x: Feature matrix of shape ``(N, D)``.
        criterion: Cosine similarity over dimension 1.
        batch_size: Rows per forward pass. ``0`` scores the whole matrix in a
            single pass, which is what the original did and what keeps results
            bit-identical to it. Set a positive value for datasets too large to
            reconstruct at once; results may then differ in the last bits,
            because BLAS can pick a different reduction order per batch shape.

    Returns:
        Scores of shape ``(N,)``, in the model's own dtype.

    Note:
        The scores are deliberately *not* promoted to float64. They carry no
        more information than the float32 model that produced them, and the
        promotion would shift the histogram bin edges by roughly float32
        epsilon — enough to move a threshold in its eighth decimal and, in
        principle, to reassign a sample sitting exactly on the boundary.
    """
    if batch_size <= 0 or batch_size >= x.shape[0]:
        single: NDArray[np.floating] = criterion(model(x), x).detach().cpu().numpy()
        return single

    chunks = [
        criterion(model(chunk), chunk).detach().cpu().numpy()
        for chunk in torch.split(x, batch_size)
    ]
    return np.concatenate(chunks)


@dataclass
class ProjectionState:
    """Everything the projection step carries between invocations.

    Attributes:
        threshold_inlier: Current inlier threshold.
        threshold_outlier: Current outlier threshold.
        threshold_outlier_tensor: The outlier threshold as a device tensor, used
            by the confidence weighting inside the negative-learning epoch.
            ``None`` when the threshold is not finite.
        inliers: The current inlier set ``L``.
        outliers: The current outlier set ``S``, unpadded.
        outlier_count: ``|S|`` as of the last projection. Read at the *start* of
            the next projection to decide whether any outlier has ever been
            found, which is what selects the symmetric first-projection rule.
        has_outliers: Whether ``S`` is non-empty.
        threshold_outlier_ema: Smoothed outlier threshold.
        cardinality_ema: Smoothed ``|S|``.
        lambda_card_ema: Smoothed ``|L| / |S|`` ratio.
        lambda_eff_ema: Smoothed loss weight.
        lambda_eff: The loss weight actually applied.
    """

    threshold_inlier: float = 0.0
    threshold_outlier: float = 0.0
    threshold_outlier_tensor: Tensor | None = None

    inliers: Tensor | None = None
    outliers: Tensor | None = None
    outlier_count: int = 0
    has_outliers: bool = False

    threshold_outlier_ema: float | None = None
    cardinality_ema: float | None = None
    lambda_card_ema: float | None = None
    lambda_eff_ema: float | None = None
    lambda_eff: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Capture the scalar state, for restoring alongside the best weights.

        Returns:
            A plain dictionary of the smoothing and threshold state.
        """
        return {
            "threshold_inlier": self.threshold_inlier,
            "threshold_outlier": self.threshold_outlier,
            "threshold_outlier_ema": self.threshold_outlier_ema,
            "cardinality_ema": self.cardinality_ema,
            "lambda_card_ema": self.lambda_card_ema,
            "lambda_eff_ema": self.lambda_eff_ema,
            "lambda_eff": self.lambda_eff,
        }

    def restore(self, snapshot: dict[str, Any], device: torch.device) -> None:
        """Reinstate a snapshot taken by :meth:`snapshot`.

        Args:
            snapshot: The dictionary to restore from.
            device: Device for the rebuilt threshold tensor.
        """
        for key, value in snapshot.items():
            setattr(self, key, value)
        self.threshold_outlier_tensor = (
            torch.tensor(self.threshold_outlier, device=device)
            if math.isfinite(self.threshold_outlier)
            else None
        )


@dataclass
class Projector:
    """Runs the projection step and owns its state.

    Attributes:
        cfg: The training schedule.
        device: Device the data lives on.
        state: Mutable projection state.
    """

    cfg: TrainingConfig
    device: torch.device
    state: ProjectionState = field(default_factory=ProjectionState)

    def project(
        self,
        model: RadonAutoencoder,
        x: Tensor,
        epoch: int,
        criterion: nn.CosineSimilarity,
        score_batch_size: int = 0,
    ) -> ProjectionRecord:
        """Re-partition ``x`` into inliers and outliers.

        Args:
            model: The model to score with.
            x: The full training matrix.
            epoch: One-based epoch number, used to select the warm-up or
                negative-learning rule and to place the curriculum ramp.
            criterion: Cosine similarity over dimension 1.
            score_batch_size: Passed through to :func:`score_dataset`.

        Returns:
            A record describing the resulting partition.
        """
        was_training = model.training
        model.eval()
        try:
            scores = score_dataset(model, x, criterion, batch_size=score_batch_size)
            record = self._project_from_scores(model, x, scores, epoch, criterion)
        finally:
            model.train(was_training)
        return record

    def _project_from_scores(
        self,
        model: RadonAutoencoder,
        x: Tensor,
        scores: NDArray[np.floating],
        epoch: int,
        criterion: nn.CosineSimilarity,
    ) -> ProjectionRecord:
        """Do the histogram work once the scores are in hand."""
        edges = histogram_bin_edges(scores, self.cfg.bin_estimator)
        counts, edges = np.histogram(scores, bins=edges)
        centers_ascending = (edges[1:] + edges[:-1]) / 2.0

        in_warmup = epoch <= self.cfg.warmup_epochs
        if in_warmup:
            thresholds = self._warmup_thresholds(counts, centers_ascending)
        else:
            thresholds = self._negative_learning_thresholds(counts, centers_ascending)

        threshold_inlier, threshold_outlier, degenerate_l, degenerate_s = thresholds

        self.state.threshold_outlier_tensor = (
            torch.tensor(threshold_outlier, device=self.device)
            if np.isfinite(threshold_outlier)
            else None
        )

        inlier_index = np.flatnonzero(scores > threshold_inlier)
        outlier_index = np.flatnonzero(scores < threshold_outlier)

        if not in_warmup:
            outlier_index = self._apply_curriculum(outlier_index, scores, epoch)

        inliers = x[torch.from_numpy(inlier_index).long().to(x.device)]
        outliers = (
            x[torch.from_numpy(outlier_index).long().to(x.device)]
            if outlier_index.size > 0
            else None
        )
        has_outliers = outliers is not None and outliers.shape[0] > 0

        self.state.threshold_inlier = threshold_inlier
        self.state.threshold_outlier = threshold_outlier
        self.state.inliers = inliers
        self.state.outliers = outliers
        self.state.has_outliers = has_outliers
        self.state.outlier_count = 0 if outliers is None else int(outliers.shape[0])

        if has_outliers:
            self._update_lambda(n_inliers=inlier_index.size, n_outliers=outlier_index.size)

        return self._build_record(
            model=model,
            criterion=criterion,
            epoch=epoch,
            scores=scores,
            n_bins=len(counts),
            threshold_inlier=threshold_inlier,
            threshold_outlier=threshold_outlier,
            degenerate_inlier=degenerate_l,
            degenerate_outlier=degenerate_s,
            n_total=int(x.shape[0]),
            inliers=inliers,
            outliers=outliers,
        )

    def _warmup_thresholds(
        self, counts: NDArray[np.int_], centers: NDArray[np.floating]
    ) -> tuple[float, float, bool, bool]:
        """Thresholds for a warm-up projection.

        Until an outlier set has ever been found, a single cut splits the data
        in two and both thresholds take that value. Once ``S`` exists, each
        threshold is estimated on its own half of the histogram.

        Args:
            counts: Histogram counts, ascending in score.
            centers: Bin centres, ascending in score.

        Returns:
            The inlier threshold, the outlier threshold, and a degeneracy flag
            for each.
        """
        never_found_outliers = self.state.outlier_count == 0

        if never_found_outliers:
            single = unimodal_threshold(counts[::-1], centers[::-1])
            return single.value, single.value, single.degenerate, single.degenerate

        inlier, outlier = self._split_halves(counts, centers)
        return inlier.value, outlier.value, inlier.degenerate, outlier.degenerate

    def _negative_learning_thresholds(
        self, counts: NDArray[np.int_], centers: NDArray[np.floating]
    ) -> tuple[float, float, bool, bool]:
        """Thresholds for a negative-learning projection.

        The outlier threshold is smoothed across projections and then clamped to
        stay strictly below the inlier threshold, so the two cuts cannot cross
        and produce an inverted partition.

        Args:
            counts: Histogram counts, ascending in score.
            centers: Bin centres, ascending in score.

        Returns:
            The inlier threshold, the outlier threshold, and a degeneracy flag
            for each.
        """
        inlier, outlier = self._split_halves(counts, centers)

        beta = self.cfg.smoothing.threshold_s
        if self.state.threshold_outlier_ema is None:
            self.state.threshold_outlier_ema = outlier.value
        else:
            self.state.threshold_outlier_ema = (
                beta * self.state.threshold_outlier_ema + (1.0 - beta) * outlier.value
            )

        threshold_outlier = self.state.threshold_outlier_ema
        if threshold_outlier >= inlier.value - THRESHOLD_MARGIN:
            threshold_outlier = inlier.value - THRESHOLD_MARGIN

        return inlier.value, threshold_outlier, inlier.degenerate, outlier.degenerate

    @staticmethod
    def _split_halves(
        counts: NDArray[np.int_], centers: NDArray[np.floating]
    ) -> tuple[UnimodalThreshold, UnimodalThreshold]:
        """Estimate one threshold per half of the score histogram.

        The inlier cut is sought in the upper half read from the top down, the
        outlier cut in the lower half read from the bottom up. Both are
        therefore presented to the triangle method with the tail on the right,
        which is the orientation it expects.

        Args:
            counts: Histogram counts, ascending in score.
            centers: Bin centres, ascending in score.

        Returns:
            The inlier and outlier threshold results.
        """
        half = len(counts) // 2
        inlier = unimodal_threshold(counts[::-1][:half], centers[::-1][:half])
        outlier = unimodal_threshold(counts[:half], centers[:half])
        return inlier, outlier

    def _apply_curriculum(
        self, outlier_index: NDArray[np.int_], scores: NDArray[np.floating], epoch: int
    ) -> NDArray[np.int_]:
        """Keep only the most confident outliers, loosening over time.

        Args:
            outlier_index: Indices currently assigned to ``S``.
            scores: Cosine scores for the whole dataset.
            epoch: Current epoch, used to place the ramp.

        Returns:
            The trimmed index array.
        """
        curriculum = self.cfg.curriculum
        if not curriculum.enabled or outlier_index.size < curriculum.min_s:
            return outlier_index

        span = max(self.cfg.epochs - self.cfg.warmup_epochs, 1)
        progress = float(np.clip((epoch - self.cfg.warmup_epochs) / span, 0.0, 1.0))
        fraction = float(
            np.clip(curriculum.start + (curriculum.end - curriculum.start) * progress, 0.0, 1.0)
        )

        outlier_scores = scores[outlier_index]
        cutoff = np.quantile(outlier_scores, fraction)
        trimmed: NDArray[np.int_] = outlier_index[outlier_scores <= (cutoff + 1e-12)]

        logger.debug(
            "Curriculum @ epoch %d: fraction=%.3f, |S| %d -> %d.",
            epoch,
            fraction,
            outlier_index.size,
            trimmed.size,
        )
        return trimmed

    def _update_lambda(self, n_inliers: int, n_outliers: int) -> None:
        """Rescale the negative-learning weight by the class imbalance.

        With far more inliers than outliers, an unweighted outlier term would be
        drowned out. The weight therefore tracks a smoothed ``|L| / |S|``,
        compressed by ``lambda_card_mode`` so that extreme ratios do not
        dominate the objective outright.

        Args:
            n_inliers: Size of ``L``.
            n_outliers: Size of ``S``.
        """
        smoothing = self.cfg.smoothing
        size = max(n_outliers, 1)

        if self.state.cardinality_ema is None:
            self.state.cardinality_ema = float(size)
        else:
            self.state.cardinality_ema = smoothing.cardinality * self.state.cardinality_ema + (
                1.0 - smoothing.cardinality
            ) * float(size)

        ratio = n_inliers / max(self.state.cardinality_ema, 1.0)

        if self.state.lambda_card_ema is None:
            self.state.lambda_card_ema = float(ratio)
        else:
            self.state.lambda_card_ema = smoothing.lambda_card * self.state.lambda_card_ema + (
                1.0 - smoothing.lambda_card
            ) * float(ratio)

        mode = self.cfg.lambda_card_mode
        compressed = self.state.lambda_card_ema
        if mode == "sqrt":
            compressed = math.sqrt(compressed)
        elif mode == "log":
            compressed = float(np.log1p(compressed))

        lambda_eff = self.cfg.lambda_s * compressed

        if self.state.lambda_eff_ema is None:
            self.state.lambda_eff_ema = float(lambda_eff)
        else:
            self.state.lambda_eff_ema = smoothing.lambda_eff * self.state.lambda_eff_ema + (
                1.0 - smoothing.lambda_eff
            ) * float(lambda_eff)

        self.state.lambda_eff = float(self.state.lambda_eff_ema)

    @torch.no_grad()
    def _build_record(
        self,
        model: RadonAutoencoder,
        criterion: nn.CosineSimilarity,
        epoch: int,
        scores: NDArray[np.floating],
        n_bins: int,
        threshold_inlier: float,
        threshold_outlier: float,
        degenerate_inlier: bool,
        degenerate_outlier: bool,
        n_total: int,
        inliers: Tensor,
        outliers: Tensor | None,
    ) -> ProjectionRecord:
        """Assemble the telemetry for this projection."""
        percentile_values = np.percentile(scores, _PERCENTILES)
        percentiles = {
            f"p{p}": float(v) for p, v in zip(_PERCENTILES, percentile_values, strict=True)
        }

        inlier_mean, inlier_std = _mean_std(
            criterion(model(inliers), inliers) if inliers.numel() > 0 else None
        )
        outlier_mean, outlier_std = _mean_std(
            criterion(model(outliers), outliers)
            if outliers is not None and outliers.numel() > 0
            else None
        )

        n_inliers = int(inliers.shape[0])
        n_outliers = 0 if outliers is None else int(outliers.shape[0])

        return ProjectionRecord(
            epoch=epoch,
            n_bins=n_bins,
            threshold_inlier=threshold_inlier,
            threshold_outlier=threshold_outlier,
            lambda_eff=self.state.lambda_eff,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_unassigned=n_total - n_inliers - n_outliers,
            degenerate_inlier_threshold=degenerate_inlier,
            degenerate_outlier_threshold=degenerate_outlier,
            score_min=float(np.min(scores)),
            score_mean=float(scores.mean()),
            score_max=float(np.max(scores)),
            score_percentiles=percentiles,
            cos_inlier_mean=inlier_mean,
            cos_inlier_std=inlier_std,
            cos_outlier_mean=outlier_mean,
            cos_outlier_std=outlier_std,
        )


def _mean_std(values: Tensor | None) -> tuple[float, float]:
    """Mean and population standard deviation of a tensor.

    Args:
        values: The tensor, or ``None`` when the set was empty.

    Returns:
        The mean and standard deviation, both NaN for an empty input.
    """
    if values is None or values.numel() == 0:
        return float("nan"), float("nan")
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def pad_outliers_to_inliers(
    outliers: Tensor | None, n_inliers: int, n_features: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Pad ``S`` with zero rows so it can be zipped with ``L`` batch by batch.

    RADON pairs each inlier with an outlier inside a single loader. The sets
    rarely have the same size, so the shorter one is padded with zeros and a
    mask records which rows are real. Padding rows are inert: cosine similarity
    against a zero vector is exactly zero under the criterion's epsilon, so they
    contribute neither loss nor gradient.

    Args:
        outliers: The outlier set, or ``None`` when empty.
        n_inliers: Target length.
        n_features: Feature count, for building zero rows.
        device: Device for the new tensors.

    Returns:
        The padded outlier matrix and a float mask of shape ``(n_inliers,)``
        holding 1 for real rows and 0 for padding.
    """
    if outliers is None or outliers.shape[0] == 0:
        return (
            torch.zeros((n_inliers, n_features), device=device),
            torch.zeros((n_inliers,), device=device),
        )

    n_outliers = outliers.shape[0]
    if n_outliers >= n_inliers:
        return outliers[:n_inliers], torch.ones((n_inliers,), device=device)

    padding = torch.zeros((n_inliers - n_outliers, n_features), device=device)
    mask = torch.cat(
        [
            torch.ones((n_outliers,), device=device),
            torch.zeros((n_inliers - n_outliers,), device=device),
        ]
    )
    return torch.cat([outliers, padding], dim=0), mask
