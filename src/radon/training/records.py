"""Structured training telemetry.

This module replaces the most fragile mechanism in the original code. There,
``fit`` printed formatted lines, a wrapper captured stdout with
``contextlib.redirect_stdout``, and roughly 120 lines of regular expressions
re-parsed those strings back into numbers to build CSV files. The numbers made
that round trip as text and could be lost silently: any change to an f-string
stopped the pattern matching, and the CSV came out with nothing but a header
and no error. Worse, the wrapper defaulted ``verbose`` to ``False``, so calling
it without explicitly passing ``verbose=True`` guaranteed an empty dump.

Here the trainer appends typed records as it goes. Logging reads from those
records; it is never the source of them.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from radon.exceptions import RadonError

__all__ = ["EpochRecord", "ProjectionRecord", "TrainingHistory"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochRecord:
    """Everything measured during one training epoch.

    Attributes:
        epoch: One-based epoch number.
        phase: ``warmup`` or ``negative_learning``.
        loss_mean: Mean total loss over the samples seen.
        loss_inlier_mean: Mean reconstruction loss on the inlier set ``L``.
        loss_outlier_mean: Mean negative-learning loss over real outliers,
            excluding padding.
        lambda_used: Effective weight applied to the outlier loss, after the
            ramp factor.
        loss_ratio: ``lambda_used * loss_outlier_mean / loss_inlier_mean``, a
            quick read on whether the two terms are balanced.
        n_outliers_seen: Number of real outlier samples contributing this epoch.
        cos_outlier_self: Mean cosine between an outlier's reconstruction and
            the outlier itself. Negative learning should drive this down.
        cos_outlier_inverse: Mean cosine between that reconstruction and the
            inverted target. Should rise as the objective takes hold.
        grad_norm: L2 norm of the gradients after the last optimisation step.
        param_norm: L2 norm of the parameters at the end of the epoch.
        duration_s: Wall-clock seconds.
        had_projection: Whether a projection ran at the end of this epoch.
        nonfinite: Whether any batch produced a non-finite loss and was skipped.
    """

    epoch: int
    phase: str
    loss_mean: float
    loss_inlier_mean: float
    loss_outlier_mean: float
    lambda_used: float
    loss_ratio: float
    n_outliers_seen: int
    cos_outlier_self: float
    cos_outlier_inverse: float
    grad_norm: float
    param_norm: float
    duration_s: float
    had_projection: bool
    nonfinite: bool

    def format_line(self) -> str:
        """Render the record as the single log line emitted per epoch.

        Returns:
            A compact, fixed-width summary.
        """
        return (
            f"Ep {self.epoch:03d} [{self.phase:<17}] | "
            f"loss={self.loss_mean:.6f} | inlier={self.loss_inlier_mean:.6f} | "
            f"outlier={self.loss_outlier_mean:.6f} | lambda={self.lambda_used:.3g} | "
            f"ratio={self.loss_ratio:.2f}x | n_out={self.n_outliers_seen} | "
            f"cos(S_hat,S)={self.cos_outlier_self:.3f} | "
            f"cos(S_hat,F(S))={self.cos_outlier_inverse:.3f} | "
            f"grad={self.grad_norm:.3e} | param={self.param_norm:.3e} | "
            f"{self.duration_s:.2f}s"
        )


@dataclass(frozen=True)
class ProjectionRecord:
    """Everything measured during one projection step.

    Attributes:
        epoch: Epoch at which the projection ran.
        n_bins: Bins chosen by the histogram estimator.
        threshold_inlier: Scores above this join ``L``.
        threshold_outlier: Scores below this join ``S``.
        lambda_eff: Loss weight derived from the new partition.
        n_inliers: Size of ``L``.
        n_outliers: Size of ``S`` after any curriculum trimming.
        n_unassigned: Samples belonging to neither set.
        degenerate_inlier_threshold: The inlier threshold landed on the edge of
            the score distribution, leaving ``L`` nearly empty.
        degenerate_outlier_threshold: The same for the outlier threshold.
        score_min: Smallest cosine score.
        score_mean: Mean cosine score.
        score_max: Largest cosine score.
        score_percentiles: Selected percentiles of the score distribution.
        cos_inlier_mean: Mean cosine between ``L`` and its reconstruction.
        cos_inlier_std: Standard deviation of the above.
        cos_outlier_mean: Mean cosine between ``S`` and its reconstruction.
        cos_outlier_std: Standard deviation of the above.
    """

    epoch: int
    n_bins: int
    threshold_inlier: float
    threshold_outlier: float
    lambda_eff: float
    n_inliers: int
    n_outliers: int
    n_unassigned: int
    degenerate_inlier_threshold: bool
    degenerate_outlier_threshold: bool
    score_min: float
    score_mean: float
    score_max: float
    score_percentiles: dict[str, float] = field(default_factory=dict)
    cos_inlier_mean: float = float("nan")
    cos_inlier_std: float = float("nan")
    cos_outlier_mean: float = float("nan")
    cos_outlier_std: float = float("nan")

    def format_lines(self) -> list[str]:
        """Render the record as the block logged after a projection.

        Returns:
            One string per line.
        """
        percentiles = " ".join(f"{k}={v:.4f}" for k, v in self.score_percentiles.items())
        return [
            f"Projection @ epoch {self.epoch}: {self.n_bins} bins",
            f"  threshold_L={self.threshold_inlier:.6f} | "
            f"threshold_S={self.threshold_outlier:.6f} | lambda_eff={self.lambda_eff:.3f}",
            f"  |L|={self.n_inliers} | |S|={self.n_outliers} | |U|={self.n_unassigned}",
            f"  scores: min={self.score_min:.4f} mean={self.score_mean:.4f} "
            f"max={self.score_max:.4f} | {percentiles}",
            f"  cos(L_hat,L)={self.cos_inlier_mean:.6f} +/- {self.cos_inlier_std:.6f} | "
            f"cos(S_hat,S)={self.cos_outlier_mean:.6f} +/- {self.cos_outlier_std:.6f}",
        ]


@dataclass
class TrainingHistory:
    """The complete telemetry of one training run.

    Attributes:
        epochs: One record per epoch, in order.
        projections: One record per projection, in order.
        stopped_early: Whether the patience ran out before the last epoch.
        best_epoch: Epoch whose weights were restored, if any.
        best_metric: Value of the monitored metric at ``best_epoch``.
    """

    epochs: list[EpochRecord] = field(default_factory=list)
    projections: list[ProjectionRecord] = field(default_factory=list)
    stopped_early: bool = False
    best_epoch: int | None = None
    best_metric: float = float("inf")

    @property
    def losses(self) -> list[float]:
        """Mean total loss per epoch, for plotting."""
        return [record.loss_mean for record in self.epochs]

    @property
    def inlier_thresholds(self) -> list[float]:
        """Inlier threshold from each projection."""
        return [record.threshold_inlier for record in self.projections]

    @property
    def outlier_thresholds(self) -> list[float]:
        """Outlier threshold from each projection."""
        return [record.threshold_outlier for record in self.projections]

    @property
    def total_duration_s(self) -> float:
        """Total wall-clock time across all epochs."""
        return sum(record.duration_s for record in self.epochs)

    def final_thresholds(self) -> tuple[float, float]:
        """The thresholds in force at the end of training.

        Returns:
            The inlier and outlier thresholds from the last projection.

        Raises:
            RadonError: If no projection ever ran, which means the schedule
                never reached a projection epoch and the model has no decision
                threshold attached to it.
        """
        if not self.projections:
            raise RadonError(
                "No projection was performed during training, so there is no decision "
                "threshold. Lower training.projection_every_warmup or raise training.epochs."
            )
        last = self.projections[-1]
        return last.threshold_inlier, last.threshold_outlier

    def write_csv(self, directory: Path, prefix: str) -> tuple[Path, Path]:
        """Write both record streams to CSV.

        Args:
            directory: Destination directory, created if absent.
            prefix: File name stem, e.g. ``mnist_rep0``.

        Returns:
            The epoch and projection file paths.

        Raises:
            RadonError: If the files cannot be written.
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            epoch_path = _write_records(
                directory / f"{prefix}_epochs.csv", self.epochs, EpochRecord
            )
            projection_path = _write_records(
                directory / f"{prefix}_projections.csv", self.projections, ProjectionRecord
            )
        except OSError as exc:
            raise RadonError(f"Cannot write training records to '{directory}': {exc}") from exc

        logger.info("Training records written to '%s' and '%s'.", epoch_path, projection_path)
        return epoch_path, projection_path


def _write_records(path: Path, records: Sequence[Any], record_type: type) -> Path:
    """Write a sequence of dataclass records to a CSV file.

    Nested dictionaries are flattened into ``parent_key`` columns so the result
    stays a flat table.

    Args:
        path: Destination file.
        records: The records to write.
        record_type: Dataclass type, used to derive the header even when
            ``records`` is empty.

    Returns:
        The path that was written.
    """
    rows = [_flatten(asdict(record)) for record in records]
    header = _header_for(rows, record_type)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})

    return path


def _header_for(rows: Iterable[dict[str, Any]], record_type: type) -> list[str]:
    """Derive CSV column names from the records, falling back to the schema.

    Args:
        rows: Flattened record dictionaries.
        record_type: Dataclass type providing the base field order.

    Returns:
        Column names, with the declared fields first and any flattened extras
        appended in first-seen order.
    """
    header = [f.name for f in fields(record_type)]
    seen = set(header)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                header.append(key)
    # Fields replaced by their flattened children carry no value of their own.
    return [name for name in header if any(name in row for row in rows) or not rows]


def _flatten(record: dict[str, Any], separator: str = "_") -> dict[str, Any]:
    """Flatten one level of nested dictionaries.

    Args:
        record: The record as a dictionary.
        separator: Joins parent and child key names.

    Returns:
        A flat dictionary.
    """
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}{separator}{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat
