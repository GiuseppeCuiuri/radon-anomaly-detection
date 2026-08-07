"""Dataset abstractions shared by every RADON data source.

The pipeline is a two-stage template: :meth:`DatasetBuilder.load` reads raw
splits from disk, :meth:`DatasetBuilder.preprocess` turns them into the
anomaly-detection protocol, and :meth:`DatasetBuilder.build` chains the two and
validates the result.

Validation uses real exceptions rather than ``assert``. The original code
guarded its shape and label invariants with bare asserts, which the interpreter
strips entirely under ``python -O`` - exactly the flag someone reaches for when
a long training run feels slow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor

from radon.exceptions import DatasetError

__all__ = [
    "AnomalyDataset",
    "DatasetBuilder",
    "LabelStats",
    "RawSplits",
    "label_statistics",
]

logger = logging.getLogger(__name__)

#: Label value for inliers.
INLIER: int = 0
#: Label value for outliers.
OUTLIER: int = 1


@dataclass(frozen=True)
class RawSplits:
    """Dataset splits as they come off disk, before the anomaly protocol.

    Labels are still in their native encoding: MNIST digits ``0..9`` for MNIST,
    binary normal/attack flags for NSL-KDD.

    Attributes:
        x_train: Training features, shape ``(N_train, D)``.
        y_train: Training labels, shape ``(N_train,)``.
        x_test: Test features, shape ``(N_test, D)``.
        y_test: Test labels, shape ``(N_test,)``.
        feature_names: Column names when the source is tabular, else ``None``.
    """

    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LabelStats:
    """Inlier/outlier counts for one split.

    Attributes:
        n_inliers: Number of samples labelled :data:`INLIER`.
        n_outliers: Number of samples labelled :data:`OUTLIER`.
        n_total: Total number of samples.
        contamination: Fraction of outliers, in ``[0, 1]``.
    """

    n_inliers: int
    n_outliers: int
    n_total: int
    contamination: float

    def __str__(self) -> str:
        return (
            f"{self.n_total} samples | inliers={self.n_inliers} "
            f"outliers={self.n_outliers} | contamination={self.contamination:.4f} "
            f"({self.contamination * 100:.2f}%)"
        )


def label_statistics(y: Tensor) -> LabelStats:
    """Summarise a binary label tensor.

    Args:
        y: Label tensor containing only :data:`INLIER` and :data:`OUTLIER`.

    Returns:
        The corresponding :class:`LabelStats`.

    Raises:
        DatasetError: If ``y`` is empty.
    """
    flat = y.reshape(-1)
    n_total = int(flat.numel())
    if n_total == 0:
        raise DatasetError("label_statistics: the label tensor is empty.")

    n_outliers = int((flat == OUTLIER).sum().item())
    n_inliers = int((flat == INLIER).sum().item())
    return LabelStats(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_total=n_total,
        contamination=n_outliers / n_total,
    )


@dataclass(frozen=True)
class AnomalyDataset:
    """A dataset already shaped into the anomaly-detection protocol.

    Features are flattened to two dimensions and labels are binary: the RADON
    autoencoder is fully connected, so a 4-D image tensor could not be consumed
    anyway. Visualisation code reshapes back using ``MnistConfig.image_shape``.

    Attributes:
        name: Dataset identifier, e.g. ``mnist``.
        x_train: Training features, shape ``(N_train, D)``, ``float32``.
        y_train: Binary training labels, shape ``(N_train,)``, ``int64``.
        x_test: Test features, shape ``(N_test, D)``, ``float32``.
        y_test: Binary test labels, shape ``(N_test,)``, ``int64``.
        feature_names: Column names for tabular sources, else ``None``.
    """

    name: str
    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    feature_names: tuple[str, ...] | None = None

    @property
    def n_features(self) -> int:
        """Input dimensionality ``D``."""
        return int(self.x_train.shape[1])

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return int(self.x_test.shape[0])

    @property
    def train_stats(self) -> LabelStats:
        """Label statistics for the training split."""
        return label_statistics(self.y_train)

    @property
    def test_stats(self) -> LabelStats:
        """Label statistics for the test split."""
        return label_statistics(self.y_test)

    def to(self, device: torch.device | str) -> AnomalyDataset:
        """Return a copy with every tensor moved to ``device``.

        Args:
            device: Target device.

        Returns:
            A new :class:`AnomalyDataset`; the original is left untouched.
        """
        return replace(
            self,
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_test=self.x_test.to(device),
            y_test=self.y_test.to(device),
        )

    def validate(self) -> None:
        """Check every invariant the training code relies on.

        Verifies rank, dtype, matching lengths, consistent dimensionality,
        binary labels, finiteness and feature-name alignment. A feature range
        outside ``[0, 1]`` is reported as a warning rather than an error,
        because the sigmoid decoder cannot reproduce such values and the
        resulting anomaly scores would be inflated.

        Raises:
            DatasetError: If any invariant is violated, with a message naming
                the offending tensor and the values observed.
        """
        for label, x, y in (
            ("train", self.x_train, self.y_train),
            ("test", self.x_test, self.y_test),
        ):
            if x.ndim != 2:
                raise DatasetError(
                    f"{self.name}: x_{label} must be 2-D (N, D), found {x.ndim}-D "
                    f"with shape {tuple(x.shape)}."
                )
            if y.ndim != 1:
                raise DatasetError(
                    f"{self.name}: y_{label} must be 1-D (N,), found {y.ndim}-D "
                    f"with shape {tuple(y.shape)}."
                )
            if x.shape[0] != y.shape[0]:
                raise DatasetError(
                    f"{self.name}: x_{label} has {x.shape[0]} samples but y_{label} "
                    f"has {y.shape[0]}."
                )
            if x.shape[0] == 0:
                raise DatasetError(f"{self.name}: the {label} split is empty.")
            if x.dtype != torch.float32:
                raise DatasetError(f"{self.name}: x_{label} must be float32, found {x.dtype}.")
            if y.dtype != torch.int64:
                raise DatasetError(f"{self.name}: y_{label} must be int64, found {y.dtype}.")
            if not torch.isfinite(x).all():
                n_bad = int((~torch.isfinite(x)).sum().item())
                raise DatasetError(
                    f"{self.name}: x_{label} contains {n_bad} non-finite value(s) (NaN or inf)."
                )

            observed = torch.unique(y)
            allowed = torch.tensor([INLIER, OUTLIER], device=y.device, dtype=y.dtype)
            if not bool(torch.isin(observed, allowed).all()):
                raise DatasetError(
                    f"{self.name}: y_{label} must contain only {{0, 1}}, found {observed.tolist()}."
                )

        if self.x_train.shape[1] != self.x_test.shape[1]:
            raise DatasetError(
                f"{self.name}: feature-count mismatch between splits - "
                f"train has {self.x_train.shape[1]}, test has {self.x_test.shape[1]}."
            )

        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise DatasetError(
                f"{self.name}: {len(self.feature_names)} feature name(s) provided but the "
                f"data has {self.n_features} column(s)."
            )

        self._warn_if_out_of_unit_range()

    def _warn_if_out_of_unit_range(self) -> None:
        """Log a warning when features fall outside ``[0, 1]``."""
        for label, x in (("train", self.x_train), ("test", self.x_test)):
            low = float(x.min().item())
            high = float(x.max().item())
            if low < 0.0 or high > 1.0:
                logger.warning(
                    "%s: x_%s spans [%.4f, %.4f], outside [0, 1]. The decoder ends in a "
                    "sigmoid, so out-of-range features cannot be reconstructed and their "
                    "anomaly scores will be inflated.",
                    self.name,
                    label,
                    low,
                    high,
                )

    def describe(self) -> str:
        """Return a one-paragraph human-readable summary.

        Returns:
            A multi-line string suitable for logging at INFO level.
        """
        return (
            f"{self.name}: {self.n_features} features\n"
            f"  train | {self.train_stats}\n"
            f"  test  | {self.test_stats}"
        )


class DatasetBuilder(ABC):
    """Template for turning a raw data source into an :class:`AnomalyDataset`.

    Subclasses implement :meth:`load` and :meth:`preprocess`; :meth:`build`
    orchestrates them and enforces the shared invariants.

    Attributes:
        name: Dataset identifier, set by each subclass.
    """

    name: ClassVar[str]

    def __init__(self, data_dir: Path, seed: int) -> None:
        """Initialise the builder.

        Args:
            data_dir: Root directory containing the dataset.
            seed: Seed for every sampling decision made during preprocessing.
        """
        self.data_dir = Path(data_dir)
        self.seed = seed

    @abstractmethod
    def load(self) -> RawSplits:
        """Read the raw splits from disk.

        Returns:
            The untouched splits with native labels.

        Raises:
            DatasetError: If the source is missing or unreadable.
        """

    @abstractmethod
    def preprocess(self, raw: RawSplits) -> AnomalyDataset:
        """Apply the anomaly-detection protocol to raw splits.

        Args:
            raw: Output of :meth:`load`.

        Returns:
            The dataset with flattened features and binary labels.

        Raises:
            DatasetError: If the protocol cannot be satisfied, for instance
                when more samples are requested than the source provides.
        """

    def build(self) -> AnomalyDataset:
        """Load, preprocess and validate in one call.

        Returns:
            A validated :class:`AnomalyDataset`.

        Raises:
            DatasetError: Propagated from any stage.
        """
        logger.info("Building dataset '%s' from '%s'.", self.name, self.data_dir)
        raw = self.load()
        dataset = self.preprocess(raw)
        dataset.validate()
        logger.info("Dataset ready.\n%s", dataset.describe())
        return dataset
